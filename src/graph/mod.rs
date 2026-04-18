//! Graph IR layer. Separate from the eager [`Tape`]: a `Graph` holds the
//! *structure* of a computation without the concrete tensor values, so that
//! compiler passes can rewrite it without incidentally computing anything.
//!
//! Flow:
//! 1. [`tracer::trace`] converts a `Tape` into a `Graph`.
//! 2. [`compile::constant_fold`] evaluates `Const + Const → Const`.
//! 3. [`compile::dead_code_elim`] removes nodes whose outputs no output uses.
//! 4. [`Graph::eval`] runs the optimized graph back out.

pub mod compile;
pub mod node;
pub mod tracer;

use std::collections::{HashMap, HashSet};

use crate::op::Op;
use crate::tape::TensorId;
use crate::tensor::Tensor;

use node::Node;

/// An IR-level handle — new numbering, independent of [`TensorId`] so that
/// compiler passes can renumber freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    /// Mapping from the tracer's source TensorIds to graph NodeIds.
    pub source_map: HashMap<TensorId, NodeId>,
    /// Which nodes are designated outputs (roots of use).
    pub outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    /// Topological order assuming nodes were appended in production order.
    /// For this crate that's always true; no Kahn's needed.
    pub fn topo(&self) -> Vec<NodeId> {
        (0..self.nodes.len()).map(NodeId).collect()
    }

    /// Return the set of NodeIds reachable from `outputs` by walking inputs.
    /// Used by DCE and for cheap printing.
    pub fn live(&self) -> HashSet<NodeId> {
        let mut live = HashSet::new();
        let mut stack: Vec<NodeId> = self.outputs.clone();
        while let Some(id) = stack.pop() {
            if !live.insert(id) {
                continue;
            }
            for &inp in &self.nodes[id.0].inputs {
                stack.push(inp);
            }
        }
        live
    }

    /// Run the graph bottom-up and return values for every live node.
    pub fn eval(&self) -> HashMap<NodeId, Tensor> {
        let live = self.live();
        let mut vals: HashMap<NodeId, Tensor> = HashMap::new();
        for id in self.topo() {
            if !live.contains(&id) {
                continue;
            }
            let node = &self.nodes[id.0];
            let out = match node.op {
                Op::Leaf | Op::Const => node
                    .constant
                    .clone()
                    .expect("Leaf/Const node missing its tensor"),
                Op::Add => bin(&vals, node, |a, b| a + b),
                Op::Sub => bin(&vals, node, |a, b| a - b),
                Op::Mul => bin(&vals, node, |a, b| a * b),
                Op::Neg => un(&vals, node, |a| -a),
                Op::Relu => un(&vals, node, |a| a.max(0.0)),
                Op::Sum => {
                    let a = &vals[&node.inputs[0]];
                    a.sum()
                }
            };
            vals.insert(id, out);
        }
        vals
    }
}

fn bin<F: Fn(f32, f32) -> f32>(vals: &HashMap<NodeId, Tensor>, node: &Node, f: F) -> Tensor {
    let a = &vals[&node.inputs[0]];
    let b = &vals[&node.inputs[1]];
    crate::tape::elementwise_binary(a, b, f)
}

fn un<F: Fn(f32) -> f32>(vals: &HashMap<NodeId, Tensor>, node: &Node, f: F) -> Tensor {
    let a = &vals[&node.inputs[0]];
    let data: Vec<f32> = a.data().iter().map(|x| f(*x)).collect();
    Tensor::new(data, a.shape().clone()).expect("un shape")
}
