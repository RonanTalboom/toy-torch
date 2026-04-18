//! v0.3 codegen tests. Verify that `emit_rust` produces plausible source for
//! a small fused recipe. The emitted code is not compiled at test time — we
//! assert on well-formed text.

use toy_torch::graph::codegen::emit_rust;
use toy_torch::graph::fusion::{Expr, FusedRecipe};

#[test]
fn emit_rust_single_input_relu() {
    // relu(in0)
    let recipe = FusedRecipe::new(Expr::Relu(Box::new(Expr::Input(0))));
    let src = emit_rust(&recipe, 1);
    assert!(src.contains("pub fn kernel(inputs: &[&[f32]], output: &mut [f32])"));
    assert!(src.contains("let in0 = inputs[0];"));
    assert!(src.contains("(in0[i].max(0.0))"));
    assert!(src.contains("for i in 0..n"));
}

#[test]
fn emit_rust_two_input_chain() {
    // (in0 - in1) * in0
    let recipe = FusedRecipe::new(Expr::Mul(
        Box::new(Expr::Sub(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Input(1)),
        )),
        Box::new(Expr::Input(0)),
    ));
    let src = emit_rust(&recipe, 2);
    assert!(src.contains("let in0 = inputs[0];"));
    assert!(src.contains("let in1 = inputs[1];"));
    assert!(src.contains("((in0[i] - in1[i]) * in0[i])"));
}

#[test]
fn emit_rust_asserts_input_count() {
    let recipe = FusedRecipe::new(Expr::Input(0));
    let src = emit_rust(&recipe, 3);
    assert!(src.contains("assert_eq!(inputs.len(), 3);"));
}
