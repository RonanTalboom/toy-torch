//! Tape → Graph conversion. Preserves op + input structure; promotes the
//! tape's materialized tensor at `Leaf`/`Const` nodes into the graph's
//! `Node::constant` slot.
//!
//! We intentionally re-evaluate ops through the graph rather than reusing the
//! tape's output tensors — that's the point of having a separate graph IR.

use crate::tape::{Tape, TensorId};

use super::{Graph, Node, NodeId};

/// Trace a tape into a graph. The resulting graph's outputs are `outputs`
/// (graph-side NodeIds corresponding to the tape's final `TensorId`s).
pub fn trace(tape: &Tape, outputs: &[TensorId]) -> Graph {
    let mut g = Graph::new();

    for (tape_idx, node) in tape.nodes().iter().enumerate() {
        let tape_id = TensorId(tape_idx);
        let new_node = match node.op {
            crate::op::Op::Leaf => Node::leaf(tape.get(node.output).clone()),
            crate::op::Op::Const => Node::constant(tape.get(node.output).clone()),
            _ => {
                let inputs: Vec<NodeId> = node
                    .inputs
                    .iter()
                    .map(|tid| {
                        *g.source_map
                            .get(tid)
                            .expect("tracer: input must have been produced earlier")
                    })
                    .collect();
                Node::op(node.op, inputs)
            }
        };
        let nid = g.push(new_node);
        g.source_map.insert(tape_id, nid);
    }

    g.outputs = outputs
        .iter()
        .map(|tid| {
            *g.source_map
                .get(tid)
                .expect("tracer: output tensor not on tape")
        })
        .collect();
    g
}
