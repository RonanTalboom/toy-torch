//! Reverse-mode autodiff over the [`Tape`].
//!
//! Walk nodes in reverse; maintain a gradient accumulator per `TensorId`. For
//! each node, add its gradient contribution to its inputs. At the end, the
//! gradient for every leaf with `requires_grad = true` is available.

use std::collections::HashMap;

use crate::op::Op;
use crate::shape::Shape;
use crate::tape::{Tape, TensorId};
use crate::tensor::Tensor;
use crate::{Error, Result};

impl Tape {
    /// Compute gradients of the scalar tensor `loss` with respect to every
    /// leaf that has `requires_grad = true`. Returns a map `TensorId -> grad`.
    ///
    /// Contract: `loss` must be a scalar. A ShapeMismatch is returned otherwise.
    pub fn backward(&self, loss: TensorId) -> HashMap<TensorId, Tensor> {
        self.try_backward(loss)
            .expect("backward failed; use try_backward for errors")
    }

    pub fn try_backward(&self, loss: TensorId) -> Result<HashMap<TensorId, Tensor>> {
        let loss_tensor = &self.tensors[loss.0];
        if !loss_tensor.is_scalar() {
            return Err(Error::BackwardOnNonScalar(
                loss_tensor.shape().dims().to_vec(),
            ));
        }

        // grads[i] = dL/d tensors[i]  (only stored for tensors in the backward cone)
        let mut grads: HashMap<TensorId, Tensor> = HashMap::new();
        grads.insert(loss, Tensor::scalar(1.0));

        // Walk nodes in reverse *production* order.
        // We index nodes by their output id's arena position — the arena order
        // already matches production order because every op appends.
        for node in self.nodes.iter().rev() {
            if !node.requires_grad {
                continue;
            }
            let go = match grads.get(&node.output) {
                Some(g) => g.clone(),
                None => continue, // not on the backward path to loss
            };

            match node.op {
                Op::Leaf | Op::Const => {
                    // Leaves/consts don't propagate further.
                }
                Op::Add => {
                    let a = node.inputs[0];
                    let b = node.inputs[1];
                    accumulate_with_unbroadcast(&mut grads, a, &go, &self.tensors[a.0]);
                    accumulate_with_unbroadcast(&mut grads, b, &go, &self.tensors[b.0]);
                }
                Op::Sub => {
                    let a = node.inputs[0];
                    let b = node.inputs[1];
                    let neg_go = scale(&go, -1.0);
                    accumulate_with_unbroadcast(&mut grads, a, &go, &self.tensors[a.0]);
                    accumulate_with_unbroadcast(&mut grads, b, &neg_go, &self.tensors[b.0]);
                }
                Op::Mul => {
                    let a = node.inputs[0];
                    let b = node.inputs[1];
                    let ta = &self.tensors[a.0];
                    let tb = &self.tensors[b.0];
                    // d/da (a*b) = b, d/db (a*b) = a; grad flows as go * other
                    let out_shape =
                        Shape::broadcast(ta.shape(), tb.shape()).expect("bcast already ok");
                    let tb_b = tb.broadcast_to(&out_shape).expect("bcast b");
                    let ta_b = ta.broadcast_to(&out_shape).expect("bcast a");
                    let ga = mul_same(&go, &tb_b);
                    let gb = mul_same(&go, &ta_b);
                    accumulate_with_unbroadcast(&mut grads, a, &ga, ta);
                    accumulate_with_unbroadcast(&mut grads, b, &gb, tb);
                }
                Op::Neg => {
                    let a = node.inputs[0];
                    let ga = scale(&go, -1.0);
                    accumulate(&mut grads, a, &ga);
                }
                Op::Relu => {
                    let a = node.inputs[0];
                    let ta = &self.tensors[a.0];
                    // derivative is indicator (a > 0); go * (a>0)
                    let data: Vec<f32> = go
                        .data()
                        .iter()
                        .zip(ta.data().iter())
                        .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                        .collect();
                    let ga = Tensor::new(data, go.shape().clone()).expect("relu bw shape");
                    accumulate(&mut grads, a, &ga);
                }
                Op::Sum => {
                    let a = node.inputs[0];
                    let ta = &self.tensors[a.0];
                    // go is scalar; broadcast back to ta.shape
                    let ga = go.broadcast_to(ta.shape()).expect("sum bw broadcast");
                    accumulate(&mut grads, a, &ga);
                }
                Op::Matmul => {
                    // Y = A @ B;  dA = dY @ B^T;  dB = A^T @ dY
                    let a = node.inputs[0];
                    let b = node.inputs[1];
                    let ta = &self.tensors[a.0];
                    let tb = &self.tensors[b.0];
                    let ga = go.matmul2d(&tb.transpose2d()).expect("dA = dY @ B^T");
                    let gb = ta.transpose2d().matmul2d(&go).expect("dB = A^T @ dY");
                    accumulate(&mut grads, a, &ga);
                    accumulate(&mut grads, b, &gb);
                }
                Op::Fused | Op::FusedSum => {
                    // Fused / FusedSum only appear on compiled graphs, not
                    // tapes. The tape's backward sees original ops.
                    unreachable!(
                        "{:?} should not appear on the tape; only in compiled graphs",
                        node.op
                    );
                }
            }
        }

        Ok(grads)
    }
}

fn scale(t: &Tensor, factor: f32) -> Tensor {
    let data: Vec<f32> = t.data().iter().map(|x| x * factor).collect();
    Tensor::new(data, t.shape().clone()).expect("scale ok")
}

fn mul_same(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "mul_same expects matching shapes");
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x * y)
        .collect();
    Tensor::new(data, a.shape().clone()).expect("mul_same ok")
}

fn accumulate(grads: &mut HashMap<TensorId, Tensor>, id: TensorId, g: &Tensor) {
    match grads.get_mut(&id) {
        Some(existing) => {
            for (e, v) in existing.data_mut().iter_mut().zip(g.data().iter()) {
                *e += *v;
            }
        }
        None => {
            grads.insert(id, g.clone());
        }
    }
}

/// Accumulate `g` into `grads[id]`, first unbroadcasting `g` back to
/// `target.shape()`. Needed when the forward op broadcast `target` to produce
/// the output whose gradient is `g`.
fn accumulate_with_unbroadcast(
    grads: &mut HashMap<TensorId, Tensor>,
    id: TensorId,
    g: &Tensor,
    target: &Tensor,
) {
    if g.shape() == target.shape() {
        accumulate(grads, id, g);
        return;
    }
    let axes = target.shape().broadcast_axes_to(g.shape());
    let reduced = g.sum_axes(&axes);
    // Reshape to exactly target.shape() — sum_axes may have dropped dims.
    let final_g = if reduced.shape() == target.shape() {
        reduced
    } else {
        reduced
            .reshape(target.shape().dims())
            .expect("unbroadcast reshape")
    };
    accumulate(grads, id, &final_g);
}
