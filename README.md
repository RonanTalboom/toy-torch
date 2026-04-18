# toy-torch

A minimal PyTorch-shaped crate in Rust. **~2300 lines**, one runtime dep
(`thiserror`) plus `rand` and `criterion` for examples / benches. Built to
teach DL compiler internals by implementing them.

Not a framework. A learning artifact. Written after a /research session on the
[NVIDIA DL Compiler Engineer role](learning/Career/nvidia-dl-compiler-learning-path.md)
concluded that flashcards are the wrong tool and *writing a compiler* is the
right one.

## What's in the box

| Layer | File | What it does |
|-------|------|--------------|
| Shape | `src/shape.rs` | Row-major strides, NumPy broadcasting, unbroadcast-axis discovery for backward |
| Tensor | `src/tensor.rs` | CPU f32, contiguous storage, `broadcast_to`, `sum_axes`, `reshape`, `transpose2d`, naive `matmul2d` |
| Op | `src/op.rs` | Shared vocabulary (Leaf/Const/Add/Sub/Mul/Neg/Relu/Matmul/Sum/Fused) for tape + graph |
| Tape | `src/tape.rs` | Arena-based autograd tape; explicit `&mut Tape` (no global, no `Rc<RefCell>`) |
| Autograd | `src/autograd.rs` | Reverse-mode `backward()`; unbroadcast-aware, including matmul (dA = dY@Bᵀ, dB = Aᵀ@dY) |
| Graph IR | `src/graph/node.rs`, `graph/mod.rs` | Separate IR from the tape — nodes carry structure, recipes, or constants |
| Tracer | `src/graph/tracer.rs` | Tape → Graph conversion |
| Compiler | `src/graph/compile.rs` | `constant_fold` + `dead_code_elim` passes |
| Fusion | `src/graph/fusion.rs` | Elementwise-chain fusion; collapses chains into `Op::Fused` nodes with `FusedRecipe` expression tree |
| Codegen | `src/graph/codegen.rs` | `emit_rust` — walks a `FusedRecipe` and emits equivalent Rust source (demonstration of what JIT/AOT codegen would do) |

## Design decisions worth surfacing

- **Explicit tape argument.** Every op is `tape.add(a, b)` not `a + b`. This
  makes the dispatch pattern visible in the type system, which is the whole
  point of the exercise. A production framework would hide the tape behind
  operator overloads — we don't, because hiding it hides the thing we want to
  learn.

- **Separate graph IR from eager tape.** They speak the same `Op` vocabulary
  but with different node representations. The tape holds realized values;
  the graph holds only structure. Compiler passes rewrite the graph without
  needing to know whether the original computation ran eagerly.

- **Unbroadcast in backward.** When a forward op broadcasts shapes, the
  gradient flowing backward is the output shape, which must be reduced back
  to the input's original shape. `accumulate_with_unbroadcast` handles this
  via `broadcast_axes_to` + `sum_axes`. Skipping this is the most common
  subtle bug in toy autograd implementations.

- **No matmul in v0.1.** Linear regression is done with `sum(x * w) + b` —
  the same math, no 2D-matmul edge cases. Matmul arrives in v0.2 when it
  earns its keep (an MLP example that can't be written with elementwise
  alone).

## Run it

```bash
cargo build
cargo test                              # 12 unit + 5 autograd + 3 fusion + 4 compile + 3 codegen + 2 matmul + 1 doctest
cargo run --release --example linreg    # SGD: y = sum(x * w) + b converges to target in ~200 epochs
cargo run --release --example mlp       # 4→8→1 MLP trains to loss 0.005 in 500 epochs via matmul + relu
cargo run --release --example compile   # 10 traced nodes → 4 after const-fold + DCE
cargo run --release --example fuse      # demonstrates tracer + fold + fusion + DCE + emit_rust on a chain
cargo bench --bench elementwise_chain   # eager_vec vs hand_fused vs fused_interp on 1K-1M sizes
```

Expected output for `linreg` (seed 42, 200 epochs):

```
epoch 199  loss=0.000079  w=[2.487, -1.295, 0.812]  b=0.399
target  w=[2.5, -1.3, 0.8]  b=0.4
```

Expected output for `compile`:

```
eager value of loss: [33.0]
traced graph: 10 nodes
after constant-fold: 10 nodes
after DCE: 4 nodes
compiled graph loss: [33.0]
  [0] Const = [11.0, 22.0, 33.0]
  [1] Leaf = [0.5, 0.5, 0.5]
  [2] Mul inputs=[NodeId(0), NodeId(1)]
  [3] Sum inputs=[NodeId(2)]
```

## What the bench says (v0.4)

Memory-bound elementwise chain `out = relu((x - y) * z) + x * k`, 1 M elements:

| Variant | Throughput | Relative |
|---------|-----------|----------|
| `eager_vec` — 5 loops with 5 Vec<f32> intermediates | 1.9 Gelem/s | 1.0× |
| `hand_fused` — single loop (what `emit_rust` generates) | **5.8 Gelem/s** | **3.0×** |
| `fused_interp` — `Graph::eval()` on fused recipe | 85 Melem/s | **0.04×** |

This is the punchline of the whole project:

- **Fusion structure alone doesn't buy speed.** Our `FusedRecipe` is a tree of
  `Expr` enum variants; evaluating it per-element pays a `match` cost that
  wipes out the memory-traffic win. The graph says "fused" but the runtime
  is still interpreting.
- **Codegen is where the win lives.** `hand_fused` is literally what
  `emit_rust` outputs — one tight `for` loop with the whole recipe inlined
  by `rustc`. That's the gap JIT (Cranelift/LLVM) or AOT (cargo+cc+dlopen)
  closes in real compilers. Inductor emits Triton; XLA emits LLVM; TVM
  emits LLVM. None of them interpret their IR.

This is why `torch.compile` is a compiler, not just a graph optimizer.

## Roadmap

### v0.5 — close the interpreter gap (next)
- Wire `emit_rust` to `cc` + runtime `dlopen`: compile fused kernels on first
  use, load as shared library, cache by recipe hash. This is the TorchInductor
  loop, transplanted.
- Alternative: Cranelift JIT for a truly self-contained binary.
- Bench again: `fused_jit` should land between `hand_fused` and faster (because
  Cranelift can do some target-specific optimization the pre-built Rust binary
  cannot).

### v0.6 — broadcast in fusion
- Fuse across broadcast boundaries (currently we materialize broadcast before
  fusion, losing the win). Requires per-input stride metadata inside the recipe.

### v0.7 — reduction fusion
- Fuse `sum(elementwise(...))` chains into one pass with a running accumulator.
  This is where the interesting fusion algorithms live (vertical + horizontal).

### v0.8 — differential benchmark suite
- Three shapes: memory-bound, compute-bound, fused-chain-bound
- Roofline plots via cached profiling runs
- A "which optimization bought how much speedup" attribution dashboard

### Deferred (possibly never)
- GPU backend. `cust` or a thin `wgpu` compute layer would teach a lot but
  triples the project scope. See `candle` / `burn` for prior art at that
  scale.
- `no_std` support. Not worth the ergonomic cost.
- FP16/BF16. The lesson is in the framework, not in numeric kernels.

## Relation to the vault

- `learning/Career/nvidia-dl-compiler-learning-path.md` — the /job-to-path
  output that scoped this project
- `learning/Quant/Solo Polymarket edge is thesis generation not signal processing.md` —
  the same discipline-first-not-tooling-first frame that pushed
  "implementation over flashcards"
- `learning/Quant/Carver's systematic trading framework repackages consensus practice rather than originating it.md` —
  reminds us that framework re-implementations teach the originators' work,
  not the re-implementer's

## License

MIT.
