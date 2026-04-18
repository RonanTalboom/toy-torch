use toy_torch::graph::compile::{constant_fold, dead_code_elim};
use toy_torch::graph::fusion::fuse_elementwise;
use toy_torch::graph::tracer::trace;
use toy_torch::op::Op;
use toy_torch::tensor::allclose;
use toy_torch::{Tape, Tensor};

#[test]
fn fusion_collapses_elementwise_chain() {
    let mut tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(&[1.0, 2.0, 3.0, 4.0], &[4]));
    let y = tape.leaf(Tensor::from_vec(&[0.5, 0.5, 0.5, 0.5], &[4]));
    let diff = tape.sub(x, y);
    let scaled = tape.mul(diff, diff);
    let activated = tape.relu(scaled);
    let _loss = tape.sum(activated);

    let out_id = _loss;
    let g = trace(&tape, &[out_id]);

    // Before fusion: Leaf, Leaf, Sub, Mul, Relu, Sum = 6 nodes
    assert_eq!(g.nodes.len(), 6);

    let g_fused = fuse_elementwise(&g);
    // After fusion: Leaf, Leaf, Fused, Sum = 4 nodes.
    // Sub+Mul+Relu collapse into one Fused node because Sum is non-fuseable
    // so the Relu output must materialize.
    let fused_count = g_fused.nodes.iter().filter(|n| n.op == Op::Fused).count();
    assert!(fused_count >= 1, "expected at least one Fused node");

    // Same result as eager.
    let vals = g_fused.eval();
    let eager_loss = tape.get(out_id).clone();
    assert!(
        allclose(&vals[&g_fused.outputs[0]], &eager_loss, 1e-6),
        "compiled != eager: {:?} vs {:?}",
        vals[&g_fused.outputs[0]].data(),
        eager_loss.data()
    );
}

#[test]
fn fusion_does_not_duplicate_multi_use_nodes() {
    // x is used twice: in x-y and in x*z. Neither consumer should absorb x.
    let mut tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]));
    let y = tape.leaf(Tensor::from_vec(&[0.5, 0.5, 0.5], &[3]));
    let z = tape.leaf(Tensor::from_vec(&[2.0, 2.0, 2.0], &[3]));
    let a = tape.sub(x, y);
    let b = tape.mul(x, z);
    let c = tape.add(a, b);
    let loss = tape.sum(c);

    let g = trace(&tape, &[loss]);
    let g_fused = fuse_elementwise(&g);
    let vals = g_fused.eval();
    let eager = tape.get(loss).clone();
    assert!(
        allclose(&vals[&g_fused.outputs[0]], &eager, 1e-6),
        "multi-use fusion broke correctness"
    );
}

#[test]
fn full_pipeline_trace_fold_fuse_dce_matches_eager() {
    let mut tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(&[1.0, -2.0, 3.0, -4.0, 5.0], &[5]));
    let c = tape.constant(Tensor::from_vec(&[10.0, 10.0, 10.0, 10.0, 10.0], &[5]));
    let k = tape.constant(Tensor::from_vec(&[2.0, 2.0, 2.0, 2.0, 2.0], &[5]));
    let c_times_k = tape.mul(c, k); // foldable to [20, ...]
    let shifted = tape.add(x, c_times_k); // x + 20
    let activated = tape.relu(shifted);
    let loss = tape.sum(activated);

    let eager = tape.get(loss).clone();

    let g = trace(&tape, &[loss]);
    let g = constant_fold(&g);
    let g = fuse_elementwise(&g);
    let g = dead_code_elim(&g);
    let vals = g.eval();
    let compiled = &vals[&g.outputs[0]];
    assert!(
        allclose(compiled, &eager, 1e-6),
        "pipeline result != eager: {:?} vs {:?}",
        compiled.data(),
        eager.data()
    );
}
