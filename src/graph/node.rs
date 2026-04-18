use crate::op::Op;
use crate::tensor::Tensor;

use super::NodeId;

/// A single IR-level computation.
///
/// For `Leaf`/`Const`, `inputs` is empty and `constant` holds the value. For
/// everything else, `inputs` are the NodeId operands and `constant` is `None`.
#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    /// Present for `Leaf`/`Const` nodes; `None` otherwise.
    pub constant: Option<Tensor>,
}

impl Node {
    pub fn leaf(t: Tensor) -> Self {
        Self {
            op: Op::Leaf,
            inputs: vec![],
            constant: Some(t),
        }
    }

    pub fn constant(t: Tensor) -> Self {
        Self {
            op: Op::Const,
            inputs: vec![],
            constant: Some(t),
        }
    }

    pub fn op(op: Op, inputs: Vec<NodeId>) -> Self {
        Self {
            op,
            inputs,
            constant: None,
        }
    }

    pub fn is_constant(&self) -> bool {
        self.op == Op::Const
    }
}
