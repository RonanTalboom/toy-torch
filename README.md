# toy-torch

A minimal PyTorch-shaped crate in Rust. **~1300 lines**, zero heavy dependencies
(only `thiserror` + `rand` for examples), built to teach DL compiler internals
by implementing them.

Not a framework. A learning artifact. Written after a /research session on the
[NVIDIA DL Compiler Engineer role](learning/Career/nvidia-dl-compiler-learning-path.md)
concluded that flashcards are the wrong tool and *writing a compiler* is the
right one.

## What's in the box

| Layer | File | What it does |
|-------|------|--------------|
| Shape | `src/shape.rs` | Row-major strides, NumPy broadcasting, unbroadcast-axis discovery for backward |
| Tensor | `src/tensor.rs` | CPU f32, contiguous storage, `broadcast_to`, `sum_axes`, `reshape` |
| Op | `src/op.rs` | The shared vocabulary (Leaf/Const/Add/Sub/Mul/Neg/Relu/Sum) used by both tape + graph |
| Tape | `src/tape.rs` | Arena-based autograd tape; explicit `&mut Tape` (no global, no `Rc<RefCell>`) |
| Autograd | `src/autograd.rs` | Reverse-mode `backward()`; handles broadcasting via `accumulate_with_unbroadcast` |
| Graph IR | `src/graph/node.rs`, `graph/mod.rs` | Separate IR from the tape — nodes carry structure only, not values |
| Tracer | `src/graph/tracer.rs` | Tape → Graph conversion |
| Compiler | `src/graph/compile.rs` | `constant_fold` + `dead_code_elim` passes |

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
cargo test                              # 10 unit + 5 autograd + 4 compile + 1 doctest
cargo run --release --example linreg    # SGD converges to target in ~200 epochs
cargo run --release --example compile   # 10 traced nodes → 4 after const-fold + DCE
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

## Roadmap

### v0.2 — fusion and matmul
- `Matmul` op with 2D-only semantics + analytic backward
- Elementwise fusion pass (merge chains of unary/binary ops into one `Fused`
  node; eval as one sweep to reduce memory traffic)
- `Mean` reduce + numerical-stable `LogSoftmax`
- MLP example — demonstrate gradient flow through relu + matmul

### v0.3 — compiler backends
- Replace the graph-interpreter `eval` with a codegen backend
- Target 1: emit Rust source that a separate `cargo build` compiles and runs
- Target 2: emit LLVM IR via `inkwell` or Cranelift for JIT compilation
- This is where the compiler curriculum's Domain 1 (codegen) pays off

### v0.4 — differential benchmarks
- Bench suite comparing eager vs compiled on three shapes: memory-bound,
  compute-bound, fused-chain-bound
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
