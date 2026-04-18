//! Demonstrates the fusion pass end-to-end:
//!
//! 1. Build an elementwise chain through the tape.
//! 2. Trace to a graph.
//! 3. Run constant-fold + fusion + DCE passes.
//! 4. Show how many nodes survived, and that the fused graph produces the
//!    same output as eager.
//! 5. Emit Rust source for the fused kernel.

use toy_torch::graph::codegen::emit_rust;
use toy_torch::graph::compile::{constant_fold, dead_code_elim};
use toy_torch::graph::fusion::fuse_elementwise;
use toy_torch::graph::tracer::trace;
use toy_torch::{Tape, Tensor};

fn main() {
    let mut tape = Tape::new();

    // An elementwise chain: output = relu((x - y) * z) + x * 2.0
    let x = tape.leaf(Tensor::from_vec(&[1.0, -2.0, 3.0, -4.0, 5.0], &[5]));
    let y = tape.leaf(Tensor::from_vec(&[0.5, 1.0, 1.5, 2.0, 2.5], &[5]));
    let z = tape.leaf(Tensor::from_vec(&[2.0, 2.0, 2.0, 2.0, 2.0], &[5]));
    let two = tape.constant(Tensor::from_vec(&[2.0, 2.0, 2.0, 2.0, 2.0], &[5]));

    let diff = tape.sub(x, y);
    let scaled = tape.mul(diff, z);
    let activated = tape.relu(scaled);
    let x2 = tape.mul(x, two);
    let out = tape.add(activated, x2);

    let eager = tape.get(out).clone();
    println!("eager : {:?}", eager.data());

    // Compile pipeline.
    let g = trace(&tape, &[out]);
    println!("traced          : {} nodes", g.nodes.len());

    let g = constant_fold(&g);
    println!("after const-fold: {} nodes", g.nodes.len());

    let g = fuse_elementwise(&g);
    println!("after fusion    : {} nodes", g.nodes.len());

    let g = dead_code_elim(&g);
    println!("after DCE       : {} nodes", g.nodes.len());

    let vals = g.eval();
    let compiled = &vals[&g.outputs[0]];
    println!("compiled: {:?}", compiled.data());

    for (i, node) in g.nodes.iter().enumerate() {
        let tag = match node.op {
            toy_torch::Op::Leaf | toy_torch::Op::Const => {
                format!("{:?}", node.constant.as_ref().unwrap().data())
            }
            toy_torch::Op::Fused => format!("Fused (external inputs = {:?})", node.inputs),
            other => format!("{:?} inputs={:?}", other, node.inputs),
        };
        println!("  [{i}] {:?}  {tag}", node.op);
    }

    // Emit Rust for the first fused node (v0.3 codegen demo).
    if let Some(fused) = g.nodes.iter().find(|n| n.op == toy_torch::Op::Fused) {
        let recipe = fused.recipe.as_ref().unwrap();
        let src = emit_rust(recipe, fused.inputs.len());
        println!("\n--- generated Rust for the fused kernel ---");
        print!("{src}");
    }
}
