//! Three-way comparison on a memory-bound elementwise chain:
//!
//!     out = relu((x - y) * z) + x * k
//!
//! **eager_vec** — plain Rust slices with one intermediate Vec<f32> per op.
//!   This is what you'd write if you didn't have any framework at all. Five
//!   allocations, five passes through memory.
//!
//! **fused_interp** — toy-torch's `Graph::eval()` on the post-fusion graph.
//!   One allocation, one pass — but the per-element work is a recursive
//!   `match` on the expression tree. This is what a naive IR interpreter
//!   pays.
//!
//! **hand_fused** — what `emit_rust` would produce: a tight `for i in 0..n`
//!   loop with the whole recipe inlined. One allocation, one pass, no
//!   per-element dispatch. The speedup of `hand_fused` over `fused_interp`
//!   is exactly the gap that JIT / AOT codegen closes in a real compiler.
//!
//! The pedagogical point: fusion *structure* alone doesn't buy speedup;
//! fusion *plus codegen* does. Inductor emits Triton for this reason. TVM
//! emits LLVM. Our emit_rust would need to be wired to cargo+cc to close
//! the gap, which is on the v0.5 roadmap.

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use toy_torch::graph::compile::{constant_fold, dead_code_elim};
use toy_torch::graph::fusion::fuse_elementwise;
use toy_torch::graph::tracer::trace;
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
    }
    group.finish();
}

criterion_group!(benches, bench_chain);
criterion_main!(benches);
