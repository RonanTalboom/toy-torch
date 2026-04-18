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
    /// Reduce everything to a scalar.
    Sum,
}

impl Op {
    pub fn is_elementwise_binary(self) -> bool {
        matches!(self, Op::Add | Op::Sub | Op::Mul)
    }

    pub fn is_elementwise_unary(self) -> bool {
        matches!(self, Op::Neg | Op::Relu)
    }
}
