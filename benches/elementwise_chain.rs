//! Four-way comparison on a memory-bound elementwise chain:
//!
//!     out = relu((x - y) * z) + x * k
//!
//! **eager_vec** — plain Rust slices with one intermediate Vec<f32> per op.
//!   Five allocations, five passes through memory.
//!
//! **fused_interp** — `Graph::eval()` on the post-fusion graph. One allocation,
//!   one pass, but per-element match dispatch on the expression tree — slow.
//!
//! **hand_fused** — a hand-rolled single loop that inlines the whole recipe.
//!   What `emit_c` generates. One allocation, one pass, no dispatch. Ceiling.
//!
//! **fused_jit** — `JitKernel::compile` emits C, shells out to `cc -O3`,
//!   dlopens the shared library, calls `kernel`. Real JIT-compiled code. The
//!   point of v0.5: close the `fused_interp` → `hand_fused` gap for real.
//!
//! Pedagogical result: fusion *structure* without codegen is a speed loss
//! vs eager. Fusion + codegen is a speed win. Real ML compilers
//! (TorchInductor, XLA, TVM) never interpret their IR — they always codegen.

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use toy_torch::graph::compile::{constant_fold, dead_code_elim};
use toy_torch::graph::fusion::fuse_elementwise;
use toy_torch::graph::jit::JitKernel;
use toy_torch::graph::tracer::trace;
use toy_torch::op::Op;
use toy_torch::{Tape, Tensor};

fn build_inputs(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 + 0.5).collect();
    let z: Vec<f32> = vec![2.0; n];
    let k: Vec<f32> = vec![0.5; n];
    (x, y, z, k)
}

fn eager_vec(x: &[f32], y: &[f32], z: &[f32], k: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut diff = vec![0.0f32; n];
    for i in 0..n {
        diff[i] = x[i] - y[i];
    }
    let mut scaled = vec![0.0f32; n];
    for i in 0..n {
        scaled[i] = diff[i] * z[i];
    }
    let mut act = vec![0.0f32; n];
    for i in 0..n {
        act[i] = scaled[i].max(0.0);
    }
    let mut xk = vec![0.0f32; n];
    for i in 0..n {
        xk[i] = x[i] * k[i];
    }
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = act[i] + xk[i];
    }
    out
}

fn hand_fused(x: &[f32], y: &[f32], z: &[f32], k: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = ((x[i] - y[i]) * z[i]).max(0.0) + x[i] * k[i];
    }
    out
}

fn build_fused_graph(n: usize) -> toy_torch::graph::Graph {
    let (xd, yd, zd, kd) = build_inputs(n);
    let mut tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(&xd, &[n]));
    let y = tape.leaf(Tensor::from_vec(&yd, &[n]));
    let z = tape.leaf(Tensor::from_vec(&zd, &[n]));
    let k = tape.leaf(Tensor::from_vec(&kd, &[n]));
    let diff = tape.sub(x, y);
    let scaled = tape.mul(diff, z);
    let act = tape.relu(scaled);
    let xk = tape.mul(x, k);
    let out = tape.add(act, xk);

    let g = trace(&tape, &[out]);
    let g = constant_fold(&g);
    let g = fuse_elementwise(&g);
    dead_code_elim(&g)
}

fn bench_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_chain");
    group.measurement_time(Duration::from_secs(2));

    for &n in &[1_024usize, 16_384, 262_144, 1_048_576] {
        group.throughput(Throughput::Elements(n as u64));

        let (xd, yd, zd, kd) = build_inputs(n);

        group.bench_with_input(BenchmarkId::new("eager_vec", n), &n, |b, &_n| {
            b.iter(|| eager_vec(&xd, &yd, &zd, &kd));
        });

        group.bench_with_input(BenchmarkId::new("hand_fused", n), &n, |b, &_n| {
            b.iter(|| hand_fused(&xd, &yd, &zd, &kd));
        });

        // Pre-compile once; measure only the eval cost.
        let g = build_fused_graph(n);
        group.bench_with_input(BenchmarkId::new("fused_interp", n), &n, |b, &_n| {
            b.iter(|| g.eval());
        });

        // JIT-compiled kernel. Pre-compile once; reuse across iterations.
        // The recipe comes from the same fused graph — find the Fused node and
        // extract its recipe + external-input count.
        let fused_node = g
            .nodes
            .iter()
            .find(|n| n.op == Op::Fused)
            .expect("bench: graph should contain a Fused node");
        let recipe = fused_node
            .recipe
            .as_ref()
            .expect("Fused node missing recipe");

        // Map fused-node external inputs back to the leaf data slices. The
        // fusion pass orders inputs by first-use in the expression. We find
        // which leaf each external input points to and present the slices in
        // that order.
        let external_slices: Vec<&[f32]> = fused_node
            .inputs
            .iter()
            .map(|&nid| {
                // Each external input points to a Leaf node holding a Tensor.
                let node = &g.nodes[nid.0];
                assert_eq!(node.op, Op::Leaf, "bench: fused external input not a leaf");
                let t = node.constant.as_ref().expect("leaf missing tensor");
                // SAFETY: graph outlives the bench loop (moved into JIT closure
                // below); slice is tied to that lifetime.
                let data_ptr = t.data().as_ptr();
                unsafe { std::slice::from_raw_parts(data_ptr, n) }
            })
            .collect();

        let jit = JitKernel::compile(recipe, external_slices.len())
            .expect("bench: JitKernel::compile failed");

        group.bench_with_input(BenchmarkId::new("fused_jit", n), &n, |b, &_n| {
            let mut out = vec![0.0f32; n];
            b.iter(|| {
                jit.call(&external_slices, &mut out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_chain);
criterion_main!(benches);
