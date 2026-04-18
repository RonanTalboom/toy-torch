use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("shape mismatch: {lhs:?} vs {rhs:?}")]
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("cannot broadcast {lhs:?} with {rhs:?}")]
    BroadcastError { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("tensor id {0} not found on tape")]
    UnknownTensor(usize),

    #[error("expected a scalar tensor for backward(), got shape {0:?}")]
    BackwardOnNonScalar(Vec<usize>),

    #[error("invalid shape: {0}")]
    InvalidShape(String),
}

pub type Result<T> = std::result::Result<T, Error>;
