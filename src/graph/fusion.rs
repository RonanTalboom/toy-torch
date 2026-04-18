//! Elementwise fusion pass.
//!
//! Walks a post-const-fold graph and collapses chains of elementwise ops
//! (Add / Sub / Mul / Neg / Relu) into a single `Fused` node whose behaviour
//! is described by a [`FusedRecipe`] expression tree. The fused kernel
//! evaluates in one sweep over the output elements — one allocation, one
//! pass over memory — which is the main mechanical benefit of ML-compiler
//! fusion.
//!
//! Fusion rules (kept deliberately small for v0.2):
//!
//! - Only fuses nodes with `Op::is_fuseable() == true`.
//! - A producer is absorbed into a consumer only if it has **exactly one
//!   consumer** in the graph — otherwise we'd duplicate its work.
//! - Leaves, Consts, Matmul, Sum, and already-Fused nodes act as *external
//!   inputs* to a fusion; we do not cross their boundary.
//! - We do not fuse across broadcasting boundaries in v0.2 — shapes must
//!   match after broadcast already (we rely on the materialized broadcast
//!   in [`crate::tape::elementwise_binary`]).

use std::collections::{HashMap, HashSet};

use crate::op::Op;
use crate::tensor::Tensor;

use super::{Graph, Node, NodeId};

/// Expression tree describing the computation of a fused kernel, in terms of
/// *external inputs* (indices into the fused node's `inputs` vec) plus
/// constants baked into the recipe.
#[derive(Debug, Clone)]
pub enum Expr {
    Input(usize),
    Const(Tensor),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Relu(Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct FusedRecipe {
    pub expr: Expr,
}

impl FusedRecipe {
    pub fn new(expr: Expr) -> Self {
        Self { expr }
    }

    /// Evaluate the recipe over the given external inputs. All inputs must
    /// share the same shape (post-broadcast); the output has that shape.
    pub fn eval(&self, inputs: &[&Tensor]) -> Tensor {
        // Determine output shape = shape of the first Input we find, or a
        // broadcast of all referenced shapes. For v0.2, the fusion pass only
        // merges nodes whose inputs already all share shape post-broadcast,
        // so the first input's shape is authoritative.
        let shape = inputs
            .first()
            .map(|t| t.shape().clone())
            .expect("FusedRecipe::eval: expected at least one external input");
        let n = shape.numel();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = eval_at(&self.expr, inputs, i);
        }
        Tensor::new(out, shape).expect("FusedRecipe::eval: shape")
    }

    /// v0.7 reduction-fusion path: evaluate the recipe and reduce to a scalar
    /// sum in one pass. No intermediate per-element buffer is allocated.
    pub fn eval_sum(&self, inputs: &[&Tensor]) -> Tensor {
        let n = inputs
            .first()
            .map(|t| t.numel())
            .expect("FusedRecipe::eval_sum: expected at least one external input");
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += eval_at(&self.expr, inputs, i);
        }
        Tensor::scalar(acc)
    }
}

fn eval_at(expr: &Expr, inputs: &[&Tensor], i: usize) -> f32 {
    match expr {
        Expr::Input(k) => inputs[*k].data()[i],
        Expr::Const(t) => {
            if t.is_scalar() {
                t.data()[0]
            } else {
                t.data()[i]
            }
        }
        Expr::Add(a, b) => eval_at(a, inputs, i) + eval_at(b, inputs, i),
        Expr::Sub(a, b) => eval_at(a, inputs, i) - eval_at(b, inputs, i),
        Expr::Mul(a, b) => eval_at(a, inputs, i) * eval_at(b, inputs, i),
        Expr::Neg(a) => -eval_at(a, inputs, i),
        Expr::Relu(a) => eval_at(a, inputs, i).max(0.0),
    }
}

/// Count how many nodes reference each node as an input.
fn use_counts(g: &Graph) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for node in &g.nodes {
        for inp in &node.inputs {
            *counts.entry(*inp).or_insert(0) += 1;
        }
    }
    for out in &g.outputs {
        *counts.entry(*out).or_insert(0) += 1;
    }
    counts
}

/// Main fusion pass. Consumes the graph, emits a new one with fused chains
/// collapsed to `Op::Fused` nodes.
pub fn fuse_elementwise(g: &Graph) -> Graph {
    let uses = use_counts(g);

    // For each node, record: either "pass-through" (becomes a node in the
    // new graph unchanged or absorbed as an input) or "fused" (its expression
    // has been absorbed into a consumer's recipe).
    //
    // We build the new graph by walking in topo order. When we hit an
    // elementwise node, we greedily absorb single-use fuseable producers into
    // its expression tree.
    let mut out = Graph {
        nodes: Vec::with_capacity(g.nodes.len()),
        source_map: HashMap::new(),
        outputs: vec![],
    };

    // Map old node id → new node id (if the node survived in the new graph).
    let mut surviving: HashMap<NodeId, NodeId> = HashMap::new();
    // Map old node id → expression + external-input nodes (if the node was
    // absorbed; the consumer will pull this). For surviving nodes this is
    // still populated so the consumer can decide whether to absorb them.
    // Each entry: (Expr, external_inputs_as_old_ids).
    let mut absorbed: HashMap<NodeId, (Expr, Vec<NodeId>)> = HashMap::new();

    for (idx, node) in g.nodes.iter().enumerate() {
        let old_id = NodeId(idx);

        match node.op {
            Op::Leaf | Op::Const | Op::Matmul | Op::Sum | Op::Fused => {
                // Cannot absorb. Always surface as a surviving node.
                let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| surviving[i]).collect();
                let new_node = Node {
                    op: node.op,
                    inputs: new_inputs,
                    constant: node.constant.clone(),
                    recipe: node.recipe.clone(),
                };
                let new_id = out.push(new_node);
                surviving.insert(old_id, new_id);
            }
            op if op.is_fuseable() => {
                let can_fuse_single_use =
                    |id: &NodeId| *uses.get(id).unwrap_or(&0) == 1 && absorbed.contains_key(id);

                // Build this node's expression from its inputs. For each input:
                //   - if single-use and fuseable → inline its recipe
                //   - else → treat as an external input (Expr::Input(k))
                let mut external_ids: Vec<NodeId> = Vec::new();
                let mut external_idx: HashMap<NodeId, usize> = HashMap::new();

                let make_expr = |inp: NodeId,
                                 absorbed: &HashMap<NodeId, (Expr, Vec<NodeId>)>,
                                 external_ids: &mut Vec<NodeId>,
                                 external_idx: &mut HashMap<NodeId, usize>|
                 -> Expr {
                    if can_fuse_single_use(&inp) {
                        let (sub_expr, sub_externals) = absorbed[&inp].clone();
                        // Merge sub's externals into our external list, rewriting Input indices.
                        let mut remap = HashMap::new();
                        for (k, e_old) in sub_externals.iter().enumerate() {
                            let new_k = *external_idx.entry(*e_old).or_insert_with(|| {
                                let idx = external_ids.len();
                                external_ids.push(*e_old);
                                idx
                            });
                            remap.insert(k, new_k);
                        }
                        rewrite_input_indices(&sub_expr, &remap)
                    } else {
                        // Not absorbed — treat as external.
                        let k = *external_idx.entry(inp).or_insert_with(|| {
                            let idx = external_ids.len();
                            external_ids.push(inp);
                            idx
                        });
                        Expr::Input(k)
                    }
                };

                let expr = match op {
                    Op::Add => Expr::Add(
                        Box::new(make_expr(
                            node.inputs[0],
                            &absorbed,
                            &mut external_ids,
                            &mut external_idx,
                        )),
                        Box::new(make_expr(
                            node.inputs[1],
                            &absorbed,
                            &mut external_ids,
                            &mut external_idx,
                        )),
                    ),
                    Op::Sub => Expr::Sub(
                        Box::new(make_expr(
                            node.inputs[0],
                            &absorbed,
                            &mut external_ids,
                            &mut external_idx,
                        )),
                        Box::new(make_expr(
                            node.inputs[1],
                            &absorbed,
                            &mut external_ids,
                            &mut external_idx,
                        )),
                    ),
                    Op::Mul => Expr::Mul(
                        Box::new(make_expr(
                            node.inputs[0],
                            &absorbed,
                            &mut external_ids,
                            &mut external_idx,
                        )),
                        Box::new(make_expr(
                            node.inputs[1],
                            &absorbed,
                            &mut external_ids,
                            &mut external_idx,
                        )),
                    ),
                    Op::Neg => Expr::Neg(Box::new(make_expr(
                        node.inputs[0],
                        &absorbed,
                        &mut external_ids,
                        &mut external_idx,
                    ))),
                    Op::Relu => Expr::Relu(Box::new(make_expr(
                        node.inputs[0],
                        &absorbed,
                        &mut external_ids,
                        &mut external_idx,
                    ))),
                    _ => unreachable!("non-fuseable op in fuseable branch"),
                };

                absorbed.insert(old_id, (expr.clone(), external_ids.clone()));

                // Decide whether to materialize this node in the output graph.
                // Materialize if:
                //   - it is an output of the graph, OR
                //   - it has >1 use (multiple consumers can't all absorb it)
                //   - it is used by a non-fuseable consumer (also requires materialization)
                let node_uses = *uses.get(&old_id).unwrap_or(&0);
                let is_output = g.outputs.contains(&old_id);
                let needs_materialization =
                    is_output || node_uses != 1 || has_non_fuseable_consumer(g, old_id);

                if needs_materialization {
                    let new_inputs: Vec<NodeId> =
                        external_ids.iter().map(|old| surviving[old]).collect();
                    let new_node = Node::fused(new_inputs, FusedRecipe::new(expr));
                    let new_id = out.push(new_node);
                    surviving.insert(old_id, new_id);
                }
                // If not materialized, the consumer node will absorb it via `absorbed`.
            }
            _ => unreachable!("unhandled op in fusion pass"),
        }
    }

    out.outputs = g.outputs.iter().map(|old| surviving[old]).collect();
    out.source_map = g
        .source_map
        .iter()
        .filter_map(|(tid, nid)| surviving.get(nid).map(|nnid| (*tid, *nnid)))
        .collect();
    out
}

/// Is any consumer of `producer` a non-fuseable op (or is the producer read
/// by multiple nodes)? If so the producer must materialize.
fn has_non_fuseable_consumer(g: &Graph, producer: NodeId) -> bool {
    let mut any = false;
    for node in &g.nodes {
        if node.inputs.contains(&producer) && !node.op.is_fuseable() {
            any = true;
            break;
        }
    }
    any
}

fn rewrite_input_indices(expr: &Expr, remap: &HashMap<usize, usize>) -> Expr {
    match expr {
        Expr::Input(k) => Expr::Input(remap[k]),
        Expr::Const(t) => Expr::Const(t.clone()),
        Expr::Add(a, b) => Expr::Add(
            Box::new(rewrite_input_indices(a, remap)),
            Box::new(rewrite_input_indices(b, remap)),
        ),
        Expr::Sub(a, b) => Expr::Sub(
            Box::new(rewrite_input_indices(a, remap)),
            Box::new(rewrite_input_indices(b, remap)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(rewrite_input_indices(a, remap)),
            Box::new(rewrite_input_indices(b, remap)),
        ),
        Expr::Neg(a) => Expr::Neg(Box::new(rewrite_input_indices(a, remap))),
        Expr::Relu(a) => Expr::Relu(Box::new(rewrite_input_indices(a, remap))),
    }
}

#[allow(dead_code)]
fn collect_inputs(expr: &Expr, out: &mut HashSet<usize>) {
    match expr {
        Expr::Input(k) => {
            out.insert(*k);
        }
        Expr::Const(_) => {}
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            collect_inputs(a, out);
            collect_inputs(b, out);
        }
        Expr::Neg(a) | Expr::Relu(a) => collect_inputs(a, out),
    }
}
