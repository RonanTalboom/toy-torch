use crate::op::Op;
use crate::tensor::Tensor;

use super::fusion::FusedRecipe;
use super::NodeId;

/// A single IR-level computation.
///
/// For `Leaf`/`Const`, `inputs` is empty and `constant` holds the value. For
/// `Fused`, `recipe` holds the expression tree that describes the collapsed
/// chain. Everything else: `inputs` are NodeId operands.
#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    /// Present for `Leaf`/`Const` nodes; `None` otherwise.
    pub constant: Option<Tensor>,
    /// Present for `Fused` nodes; `None` otherwise. Describes the merged
    /// computation as an expression tree over external inputs.
    pub recipe: Option<FusedRecipe>,
}

impl Node {
    pub fn leaf(t: Tensor) -> Self {
        Self {
            op: Op::Leaf,
            inputs: vec![],
            constant: Some(t),
            recipe: None,
        }
    }

    pub fn constant(t: Tensor) -> Self {
        Self {
            op: Op::Const,
            inputs: vec![],
            constant: Some(t),
            recipe: None,
        }
    }

    pub fn op(op: Op, inputs: Vec<NodeId>) -> Self {
        Self {
            op,
            inputs,
            constant: None,
            recipe: None,
        }
    }

    pub fn fused(inputs: Vec<NodeId>, recipe: FusedRecipe) -> Self {
        Self {
            op: Op::Fused,
            inputs,
            constant: None,
            recipe: Some(recipe),
        }
    }

    pub fn is_constant(&self) -> bool {
        self.op == Op::Const
    }
}
