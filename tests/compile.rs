use toy_torch::graph::compile::{constant_fold, dead_code_elim};
use toy_torch::graph::tracer::trace;
use toy_torch::op::Op;
use toy_torch::tensor::allclose;
use toy_torch::{Tape, Tensor};

#[test]
fn constant_fold_collapses_const_chain() {
    let mut tape = Tape::new();
    let a = tape.constant(Tensor::from_vec(&[1.0, 2.0], &[2]));
    let b = tape.constant(Tensor::from_vec(&[3.0, 4.0], &[2]));
    let c = tape.mul(a, b); // [3, 8]
    let d = tape.add(c, a); // [4, 10]
    let loss = tape.sum(d); // 14

    let g = trace(&tape, &[loss]);
    let g_folded = constant_fold(&g);

    // After folding, everything collapses to a single Const for the output.
    let eval = g_folded.eval();
    let out = &eval[&g_folded.outputs[0]];
    assert!(allclose(out, &Tensor::scalar(14.0), 1e-6));

    // And the output node should be Const (folded).
    let out_node = &g_folded.nodes[g_folded.outputs[0].0];
    assert_eq!(out_node.op, Op::Const);
}

#[test]
fn constant_fold_preserves_leaf_dependency() {
    let mut tape = Tape::new();
    let a = tape.constant(Tensor::from_vec(&[1.0, 2.0], &[2]));
    let b = tape.constant(Tensor::from_vec(&[3.0, 4.0], &[2]));
    let c = tape.add(a, b); // fold to [4, 6]
    let w = tape.leaf(Tensor::from_vec(&[0.5, 0.5], &[2]));
    let y = tape.mul(c, w); // cannot fold (w is a leaf parameter)
    let loss = tape.sum(y);

    let g = trace(&tape, &[loss]);
    let g_folded = constant_fold(&g);

    // We still get the right answer.
    let eval = g_folded.eval();
    let out = &eval[&g_folded.outputs[0]];
    // (1+3)*0.5 + (2+4)*0.5 = 2 + 3 = 5
    assert!(allclose(out, &Tensor::scalar(5.0), 1e-6));

    // And the Mul should not have been folded (one input is Leaf).
    assert!(g_folded.nodes.iter().any(|n| n.op == Op::Mul));
}

#[test]
fn dce_drops_unused_chain() {
    let mut tape = Tape::new();
    let a = tape.constant(Tensor::from_vec(&[1.0, 2.0], &[2]));
    let b = tape.constant(Tensor::from_vec(&[3.0, 4.0], &[2]));
    let _dead1 = tape.add(a, b);
    let _dead2 = tape.mul(a, b);

    let keeper = tape.constant(Tensor::scalar(7.0));
    let loss = tape.sum(keeper);

    let g = trace(&tape, &[loss]);
    let g_dce = dead_code_elim(&g);

    // Should keep only the keeper + Sum on the live path.
    // (The `a` and `b` nodes are unreachable from the output.)
    let ops: Vec<Op> = g_dce.nodes.iter().map(|n| n.op).collect();
    assert!(ops.contains(&Op::Const));
    assert!(ops.contains(&Op::Sum));
    assert!(!ops.contains(&Op::Add));
    assert!(!ops.contains(&Op::Mul));
}

#[test]
fn eager_and_compiled_agree() {
    // Same computation through eager tape vs traced+optimized graph.
    let mut tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]));
    let k = tape.constant(Tensor::scalar(2.0));
    let y = tape.mul(x, k);
    let r = tape.relu(y);
    let loss = tape.sum(r);

    let eager_val = tape.get(loss).clone();

    let g = trace(&tape, &[loss]);
    let g_opt = dead_code_elim(&constant_fold(&g));
    let vals = g_opt.eval();
    let compiled = &vals[&g_opt.outputs[0]];

    assert!(
        allclose(&eager_val, compiled, 1e-6),
        "eager={:?} vs compiled={:?}",
        eager_val.data(),
        compiled.data()
    );
}
