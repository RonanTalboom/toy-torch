//! Trace a Tape into a Graph, run constant-fold and DCE passes, and evaluate
//! the compiled graph. Demonstrates that the compiled form produces the same
//! numerical result while containing fewer nodes.

use toy_torch::graph::compile::{constant_fold, dead_code_elim};
use toy_torch::graph::tracer::trace;
use toy_torch::{Tape, Tensor};

fn main() {
    let mut tape = Tape::new();

    // Constants that *should* fold.
    let a = tape.constant(Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]));
    let b = tape.constant(Tensor::from_vec(&[10.0, 10.0, 10.0], &[3]));
    let c = tape.mul(a, b); // const — should fold to [10, 20, 30]
    let d = tape.add(c, a); // const — should fold to [11, 22, 33]

    // A parameter that prevents further folding past this point.
    let w = tape.leaf(Tensor::from_vec(&[0.5, 0.5, 0.5], &[3]));
    let e = tape.mul(d, w); // not foldable (w is Leaf, not Const)
    let loss = tape.sum(e);

    // An unused computation that DCE should drop.
    let unused_a = tape.constant(Tensor::scalar(100.0));
    let unused_b = tape.constant(Tensor::scalar(200.0));
    let _dead = tape.add(unused_a, unused_b);

    println!("eager value of loss: {:?}", tape.get(loss).data());

    let g = trace(&tape, &[loss]);
    println!("traced graph: {} nodes", g.nodes.len());

    let g_folded = constant_fold(&g);
    println!("after constant-fold: {} nodes", g_folded.nodes.len());

    let g_opt = dead_code_elim(&g_folded);
    println!("after DCE: {} nodes", g_opt.nodes.len());

    let vals = g_opt.eval();
    let out_id = g_opt.outputs[0];
    println!("compiled graph loss: {:?}", vals[&out_id].data());

    // Print each node to show what survived.
    for (i, node) in g_opt.nodes.iter().enumerate() {
        let tag = match node.constant.as_ref() {
            Some(t) => format!("{:?} = {:?}", node.op, t.data()),
            None => format!("{:?} inputs={:?}", node.op, node.inputs),
        };
        println!("  [{i}] {tag}");
    }
}
