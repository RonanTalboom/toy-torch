//! Shape + stride math. Row-major (C-contiguous) layout.
//!
//! Broadcasting follows NumPy rules: shapes are right-aligned; dimensions of
//! size 1 are stretched; missing dimensions are treated as size 1.

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1 // scalar
        } else {
            self.dims.iter().product()
        }
    }

    /// Row-major (C-contiguous) strides for this shape.
    pub fn contiguous_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.rank()];
        for i in (0..self.rank().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Broadcast two shapes per NumPy rules. Returns the broadcasted shape.
    pub fn broadcast(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
        let la = lhs.rank();
        let lb = rhs.rank();
        let out_rank = la.max(lb);
        let mut out = vec![0usize; out_rank];
        for i in 0..out_rank {
            let da = if i < out_rank - la {
                1
            } else {
                lhs.dims[i - (out_rank - la)]
            };
            let db = if i < out_rank - lb {
                1
            } else {
                rhs.dims[i - (out_rank - lb)]
            };
            if da == db {
                out[i] = da;
            } else if da == 1 {
                out[i] = db;
            } else if db == 1 {
                out[i] = da;
            } else {
                return Err(Error::BroadcastError {
                    lhs: lhs.dims.clone(),
                    rhs: rhs.dims.clone(),
                });
            }
        }
        Ok(Shape::new(&out))
    }

    /// Which axes in `self` were broadcast to reach `target`? Used in backward to
    /// reduce the incoming gradient back to the input's shape.
    pub fn broadcast_axes_to(&self, target: &Shape) -> Vec<usize> {
        let src = &self.dims;
        let dst = &target.dims;
        let ra = src.len();
        let rb = dst.len();
        let mut axes = Vec::new();
        // Axes added by rank expansion
        for i in 0..(rb - ra) {
            axes.push(i);
        }
        // Axes where self had size 1 and target had size > 1
        for i in 0..ra {
            let dst_idx = i + (rb - ra);
            if src[i] == 1 && dst[dst_idx] != 1 {
                axes.push(dst_idx);
            }
        }
        axes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_basic() {
        assert_eq!(Shape::new(&[2, 3]).contiguous_strides(), vec![3, 1]);
        assert_eq!(Shape::new(&[4]).contiguous_strides(), vec![1]);
        assert_eq!(Shape::new(&[2, 3, 4]).contiguous_strides(), vec![12, 4, 1]);
    }

    #[test]
    fn broadcast_same() {
        let a = Shape::new(&[3, 4]);
        let b = Shape::new(&[3, 4]);
        assert_eq!(Shape::broadcast(&a, &b).unwrap(), a);
    }

    #[test]
    fn broadcast_scalar() {
        let a = Shape::new(&[3, 4]);
        let b = Shape::scalar();
        assert_eq!(Shape::broadcast(&a, &b).unwrap(), a);
    }

    #[test]
    fn broadcast_row_vector() {
        let a = Shape::new(&[3, 4]);
        let b = Shape::new(&[4]);
        assert_eq!(Shape::broadcast(&a, &b).unwrap(), a);
    }

    #[test]
    fn broadcast_error() {
        let a = Shape::new(&[3, 4]);
        let b = Shape::new(&[3, 5]);
        assert!(Shape::broadcast(&a, &b).is_err());
    }

    #[test]
    fn broadcast_axes() {
        let src = Shape::new(&[4]);
        let dst = Shape::new(&[3, 4]);
        assert_eq!(src.broadcast_axes_to(&dst), vec![0]);

        let src = Shape::new(&[1, 4]);
        let dst = Shape::new(&[3, 4]);
        assert_eq!(src.broadcast_axes_to(&dst), vec![0]);
    }
}
