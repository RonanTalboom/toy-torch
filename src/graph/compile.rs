//! Two minimal compiler passes over the graph IR:
//!
//! 1. [`constant_fold`] — evaluate any op whose inputs are all `Const` and
//!    replace it with a `Const` node holding the computed value.
//! 2. [`dead_code_elim`] — drop nodes not reachable from any output.
//!
//! Both return new `Graph` values (immutable pass style).

use std::collections::HashMap;

use crate::op::Op;
use crate::tape::elementwise_binary;
use crate::tensor::Tensor;

use super::{Graph, Node, NodeId};

/// Replace any op whose inputs are all `Const` nodes with a `Const` holding
/// the computed value. Repeats until no more folds are possible (runs to
/// fixpoint — our small graphs make this cheap).
pub fn constant_fold(g: &Graph) -> Graph {
    let mut out = Graph {
        nodes: Vec::with_capacity(g.nodes.len()),
        source_map: HashMap::new(),
        outputs: vec![],
    };
    // Rewrite map: old NodeId → new NodeId
    let mut rewrite: HashMap<NodeId, NodeId> = HashMap::new();

    for (idx, node) in g.nodes.iter().enumerate() {
        let old_id = NodeId(idx);
        let rewritten_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| *rewrite.get(i).expect("topo: input already rewritten"))
            .collect();

        let all_inputs_const = !rewritten_inputs.is_empty()
            && rewritten_inputs
                .iter()
                .all(|nid| out.nodes[nid.0].is_constant());

        let new_node = if all_inputs_const && is_foldable(node.op) {
            let val = eval_const(&out, node.op, &rewritten_inputs);
            Node::constant(val)
        } else {
            match node.op {
                Op::Leaf | Op::Const => node.clone(),
                _ => Node::op(node.op, rewritten_inputs),
            }
        };

        let new_id = out.push(new_node);
        rewrite.insert(old_id, new_id);
    }

    // Rewrite outputs
    out.outputs = g.outputs.iter().map(|nid| rewrite[nid]).collect();
    // Rewrite source_map (tape→graph, not strictly needed after compile but keep for debug)
    out.source_map = g
        .source_map
        .iter()
        .map(|(tid, nid)| (*tid, rewrite[nid]))
        .collect();
    out
}

fn is_foldable(op: Op) -> bool {
    matches!(
        op,
        Op::Add | Op::Sub | Op::Mul | Op::Neg | Op::Relu | Op::Sum
    )
}

fn eval_const(g: &Graph, op: Op, inputs: &[NodeId]) -> Tensor {
    let as_tensor = |id: NodeId| -> Tensor {
        g.nodes[id.0]
            .constant
            .clone()
            .expect("eval_const: non-constant input")
    };
    match op {
        Op::Add => elementwise_binary(&as_tensor(inputs[0]), &as_tensor(inputs[1]), |a, b| a + b),
        Op::Sub => elementwise_binary(&as_tensor(inputs[0]), &as_tensor(inputs[1]), |a, b| a - b),
        Op::Mul => elementwise_binary(&as_tensor(inputs[0]), &as_tensor(inputs[1]), |a, b| a * b),
        Op::Neg => {
            let t = as_tensor(inputs[0]);
            let data: Vec<f32> = t.data().iter().map(|x| -x).collect();
            Tensor::new(data, t.shape().clone()).expect("neg fold")
        }
        Op::Relu => {
            let t = as_tensor(inputs[0]);
            let data: Vec<f32> = t.data().iter().map(|x| x.max(0.0)).collect();
            Tensor::new(data, t.shape().clone()).expect("relu fold")
        }
        Op::Sum => as_tensor(inputs[0]).sum(),
        Op::Leaf | Op::Const | Op::Matmul | Op::Fused | Op::FusedSum => {
            unreachable!("op {:?} not in is_foldable set", op)
        }
    }
}

/// Drop nodes not reachable from `g.outputs`. Renumbers IDs.
pub fn dead_code_elim(g: &Graph) -> Graph {
    let live = g.live();
    let mut out = Graph {
        nodes: Vec::new(),
        source_map: HashMap::new(),
        outputs: vec![],
    };
    let mut rewrite: HashMap<NodeId, NodeId> = HashMap::new();
    for (idx, node) in g.nodes.iter().enumerate() {
        let old_id = NodeId(idx);
        if !live.contains(&old_id) {
            continue;
        }
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|nid| rewrite[nid]).collect();
        let new_node = Node {
            op: node.op,
            inputs: new_inputs,
            constant: node.constant.clone(),
            recipe: node.recipe.clone(),
        };
        let new_id = out.push(new_node);
        rewrite.insert(old_id, new_id);
    }
    out.outputs = g.outputs.iter().map(|nid| rewrite[nid]).collect();
    out.source_map = g
        .source_map
        .iter()
        .filter_map(|(tid, nid)| rewrite.get(nid).map(|nnid| (*tid, *nnid)))
        .collect();
    out
}
