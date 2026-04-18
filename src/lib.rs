//! `toy-torch` — a minimal PyTorch-shaped crate for learning DL compiler internals.
//!
//! ## Layers
//! - [`tensor`]: CPU f32 tensors with strided views and NumPy-style broadcasting
//! - [`tape`]: arena-based autograd tape ([`Tape`] owns all tensors by [`TensorId`])
//! - [`op`]: the `Op` enum — the set of primitives the tape and graph both understand
//! - [`autograd`]: reverse-mode backward, walking the tape top-down
//! - [`graph`]: separate graph IR + tracer + compiler passes (const-fold, DCE)
//!
//! ## Shape of the API (minimal example)
//! ```no_run
//! use toy_torch::{Tape, Tensor};
//! let mut tape = Tape::new();
//! let x = tape.leaf(Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]));
//! let w = tape.leaf(Tensor::from_vec(&[0.5, -0.5, 0.25], &[3]));
//! let y = tape.mul(x, w);
//! let loss = tape.sum(y);
//! let grads = tape.backward(loss);
//! println!("dL/dw = {:?}", grads.get(&w));
//! ```
//!
//! ## What this is NOT
//! Production anything. No GPU, no SIMD, no BLAS, no fp16, no distributed,
//! no fused kernels. Ship artifact, not runtime.

pub mod autograd;
pub mod error;
pub mod graph;
pub mod op;
pub mod shape;
pub mod tape;
pub mod tensor;

pub use error::{Error, Result};
pub use op::Op;
pub use shape::Shape;
pub use tape::{Tape, TensorId};
pub use tensor::Tensor;
