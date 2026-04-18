//! The shared op vocabulary. The eager tape and the captured graph both speak
//! in `Op` + `inputs`. Splitting op-definition from backend-dispatch keeps the
//! compiler passes unaware of whether a tensor was produced eagerly or
//! symbolically.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// Source of a value; not derived from anything.
    Leaf,
    /// Compile-time constant (held separately so constant-fold can collapse chains).
    Const,
    Add,
    Sub,
    Mul,
    Neg,
    Relu,
    /// 2D matrix multiplication: `[M, K] @ [K, N] -> [M, N]`.
    Matmul,
    /// Reduce everything to a scalar.
    Sum,
    /// A fused chain of elementwise ops, eval'd in one sweep. Produced by the
    /// fusion compiler pass. The associated recipe lives in
    /// [`crate::graph::fusion::FusedRecipe`] and is stored on the graph node.
    Fused,
}

impl Op {
    pub fn is_elementwise_binary(self) -> bool {
        matches!(self, Op::Add | Op::Sub | Op::Mul)
    }

    pub fn is_elementwise_unary(self) -> bool {
        matches!(self, Op::Neg | Op::Relu)
    }

    /// Fuseable = pure elementwise, same-shape-after-broadcast. Matmul and Sum
    /// are not fuseable under this simple scheme.
    pub fn is_fuseable(self) -> bool {
        self.is_elementwise_binary() || self.is_elementwise_unary()
    }
}
