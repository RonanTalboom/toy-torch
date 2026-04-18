//! Reduction fusion (v0.7).
//!
//! Detects the pattern `Sum ← Fused` in the graph and collapses it into
//! a single `FusedSum` node. In one pass over the output elements, the
//! fused kernel evaluates the recipe and accumulates the running sum —
//! no intermediate per-element buffer is needed.
//!
//! This mirrors the reduction-fusion optimization in TorchInductor: a
//! chain like `sum(relu(x - y) * z)` compiles to a single kernel that
//! walks the inputs once, computing the element value and folding it
//! into the accumulator.
//!
//! The companion pass adds a new [`Op`] variant (`Op::FusedSum`) whose
//! recipe is reused from `FusedRecipe`. Eval returns a scalar; emit_c
//! produces a `float acc = 0; for(i){ acc += recipe(i); } *output = acc;`
//! kernel.

use std::collections::HashMap;

use crate::op::Op;

use super::{Graph, Node, NodeId};

/// Fuse any `Sum(Fused(...))` chain into `FusedSum(...)`.
///
/// Rules:
/// - The Sum's single input must be a Fused node.
/// - The Fused node must have exactly one consumer (the Sum) — otherwise we
///   can't absorb it without duplicating work elsewhere.
/// - Both nodes collapse into one `FusedSum` node inheriting the recipe and
///   external inputs from the Fused node.
pub fn fuse_reductions(g: &Graph) -> Graph {
    // Count uses so we only absorb single-use producers.
    let mut uses: HashMap<NodeId, usize> = HashMap::new();
    for node in &g.nodes {
        for inp in &node.inputs {
            *uses.entry(*inp).or_insert(0) += 1;
        }
    }
    for out in &g.outputs {
        *uses.entry(*out).or_insert(0) += 1;
    }

    let mut out = Graph {
        nodes: Vec::with_capacity(g.nodes.len()),
        source_map: HashMap::new(),
        outputs: vec![],
    };
    let mut rewrite: HashMap<NodeId, NodeId> = HashMap::new();
    let mut absorbed: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

    for (idx, node) in g.nodes.iter().enumerate() {
        let old_id = NodeId(idx);

        if absorbed.contains(&old_id) {
            // Already folded into a FusedSum. Skip.
            continue;
        }

        // Is this a Sum(Fused(single-use))? If yes, absorb and emit FusedSum.
        if node.op == Op::Sum && node.inputs.len() == 1 {
            let inp_old = node.inputs[0];
            let inp_node = &g.nodes[inp_old.0];
            let inp_single_use = uses.get(&inp_old).copied().unwrap_or(0) == 1;
            if inp_node.op == Op::Fused && inp_single_use {
                // Build a FusedSum whose external inputs are the Fused node's
                // (remapped to the new graph's NodeIds).
                let new_inputs: Vec<NodeId> =
                    inp_node.inputs.iter().map(|nid| rewrite[nid]).collect();
                let recipe = inp_node.recipe.clone().expect("Fused node missing recipe");
                let mut new_node = Node::op(Op::FusedSum, new_inputs);
                new_node.recipe = Some(recipe);
                let new_id = out.push(new_node);
                rewrite.insert(old_id, new_id);
                absorbed.insert(inp_old);
                continue;
            }
        }

        // Default: copy the node over with remapped inputs.
        let new_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| {
                *rewrite
                    .get(i)
                    .expect("reduction-fuse: input should already be rewritten")
            })
            .collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compile::{constant_fold, dead_code_elim};
    use crate::graph::fusion::fuse_elementwise;
    use crate::graph::tracer::trace;
    use crate::tensor::allclose;
    use crate::{Tape, Tensor};

    #[test]
    fn fuse_sum_of_fused() {
        let mut tape = Tape::new();
        let x = tape.leaf(Tensor::from_vec(&[1.0, -2.0, 3.0, -4.0, 5.0], &[5]));
        let y = tape.leaf(Tensor::from_vec(&[0.5, 0.5, 0.5, 0.5, 0.5], &[5]));
        let diff = tape.sub(x, y);
        let rel = tape.relu(diff);
        let loss = tape.sum(rel);

        let eager = tape.get(loss).clone();

        let g = trace(&tape, &[loss]);
        let g = constant_fold(&g);
        let g = fuse_elementwise(&g);
        let g = fuse_reductions(&g);
        let g = dead_code_elim(&g);

        // After reduction fusion: Sum + Fused collapse into one FusedSum.
        let has_fused_sum = g.nodes.iter().any(|n| n.op == Op::FusedSum);
        assert!(
            has_fused_sum,
            "expected a FusedSum node after reduction fusion"
        );

        let vals = g.eval();
        let compiled = &vals[&g.outputs[0]];
        assert!(
            allclose(compiled, &eager, 1e-6),
            "FusedSum result {:?} != eager {:?}",
            compiled.data(),
            eager.data()
        );
    }

    #[test]
    fn fusion_preserves_multi_use_fused_nodes() {
        // A Fused node consumed by both Sum and a Neg can't be absorbed.
        let mut tape = Tape::new();
        let x = tape.leaf(Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]));
        let y = tape.leaf(Tensor::from_vec(&[0.5, 0.5, 0.5], &[3]));
        let diff = tape.sub(x, y); // fuseable
        let s = tape.sum(diff);
        let n = tape.neg(diff);
        let m = tape.sum(n);
        let total = tape.add(s, m);

        let g = trace(&tape, &[total]);
        let g = constant_fold(&g);
        let g = fuse_elementwise(&g);
        let g_reduced = fuse_reductions(&g);

        // The fused (sub) node is consumed by two paths; should NOT be
        // absorbed into FusedSum on either path (the first Sum could try but
        // we gate on single-use). Verify correctness regardless.
        let vals = g_reduced.eval();
        let eager = tape.get(total).clone();
        let compiled = &vals[&g_reduced.outputs[0]];
        assert!(
            allclose(compiled, &eager, 1e-6),
            "multi-use case broke correctness"
        );
    }
}
