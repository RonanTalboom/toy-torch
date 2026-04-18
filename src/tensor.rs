//! CPU f32 tensors with a contiguous backing store. Views are implicit via
//! `shape` — this crate does not support non-contiguous strides for v0.1 (the
//! extra complexity distracts from the teaching goal).
//!
//! Broadcasting works by *materializing* broadcasted values into a new tensor.
//! A production framework would avoid this with strided views; we don't, to
//! keep the code obvious.

use crate::shape::Shape;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Shape) -> Result<Self> {
        let expected = shape.numel();
        if data.len() != expected {
            return Err(Error::InvalidShape(format!(
                "data.len() = {}, shape numel = {} ({:?})",
                data.len(),
                expected,
                shape.dims()
            )));
        }
        Ok(Self { data, shape })
    }

    pub fn from_vec(data: &[f32], dims: &[usize]) -> Self {
        Self::new(data.to_vec(), Shape::new(dims)).expect("from_vec: length mismatch with dims")
    }

    pub fn scalar(value: f32) -> Self {
        Self {
            data: vec![value],
            shape: Shape::scalar(),
        }
    }

    pub fn zeros(dims: &[usize]) -> Self {
        let shape = Shape::new(dims);
        let n = shape.numel();
        Self {
            data: vec![0.0; n],
            shape,
        }
    }

    pub fn ones(dims: &[usize]) -> Self {
        let shape = Shape::new(dims);
        let n = shape.numel();
        Self {
            data: vec![1.0; n],
            shape,
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.rank() == 0 || self.data.len() == 1
    }

    /// Broadcast `self` to `target` shape by materializing.
    pub fn broadcast_to(&self, target: &Shape) -> Result<Tensor> {
        if self.shape == *target {
            return Ok(self.clone());
        }
        let _ = Shape::broadcast(&self.shape, target)?; // validate

        let out_numel = target.numel();
        let mut out = vec![0.0f32; out_numel];
        let out_strides = target.contiguous_strides();

        let src_dims = self.shape.dims();
        let dst_dims = target.dims();
        let r_diff = dst_dims.len() - src_dims.len();
        let src_strides = self.shape.contiguous_strides();

        for i in 0..out_numel {
            // Unravel linear index i into per-dim indices in target.
            let mut rem = i;
            let mut src_idx = 0usize;
            for axis in 0..dst_dims.len() {
                let coord = rem / out_strides[axis];
                rem %= out_strides[axis];
                if axis >= r_diff {
                    let src_axis = axis - r_diff;
                    // If source dim is 1, broadcast — coord contributes 0 stride.
                    if src_dims[src_axis] != 1 {
                        src_idx += coord * src_strides[src_axis];
                    }
                }
            }
            out[i] = self.data[src_idx];
        }
        Tensor::new(out, target.clone())
    }

    /// Sum over specific axes, keeping other dimensions. Used in the backward
    /// pass to "unbroadcast" a gradient back to the input's shape.
    pub fn sum_axes(&self, axes: &[usize]) -> Tensor {
        if axes.is_empty() {
            return self.clone();
        }
        let dims = self.shape.dims();
        let rank = dims.len();
        let keep: Vec<bool> = (0..rank).map(|i| !axes.contains(&i)).collect();
        let out_dims: Vec<usize> = (0..rank).filter(|&i| keep[i]).map(|i| dims[i]).collect();
        let out_shape = Shape::new(&out_dims);
        let mut out = vec![0.0f32; out_shape.numel().max(1)];
        let in_strides = self.shape.contiguous_strides();
        let out_strides = out_shape.contiguous_strides();

        for i in 0..self.numel() {
            let mut rem = i;
            let mut out_idx = 0usize;
            let mut kept_axis = 0;
            for axis in 0..rank {
                let coord = rem / in_strides[axis];
                rem %= in_strides[axis];
                if keep[axis] {
                    if !out_strides.is_empty() {
                        out_idx += coord * out_strides[kept_axis];
                    }
                    kept_axis += 1;
                }
            }
            out[out_idx] += self.data[i];
        }

        Tensor::new(out, out_shape).expect("sum_axes: internal shape mismatch")
    }

    /// Sum all elements → scalar.
    pub fn sum(&self) -> Tensor {
        let total: f32 = self.data.iter().sum();
        Tensor::scalar(total)
    }

    /// Reshape without moving data (must match numel).
    pub fn reshape(&self, dims: &[usize]) -> Result<Tensor> {
        let new_shape = Shape::new(dims);
        if new_shape.numel() != self.numel() {
            return Err(Error::ShapeMismatch {
                lhs: self.shape.dims().to_vec(),
                rhs: dims.to_vec(),
            });
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
        })
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

/// Check two tensors are element-wise close.
pub fn allclose(a: &Tensor, b: &Tensor, tol: f32) -> bool {
    if a.shape != b.shape {
        return false;
    }
    a.data
        .iter()
        .zip(b.data.iter())
        .all(|(x, y)| (x - y).abs() < tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_scalar_to_matrix() {
        let s = Tensor::scalar(3.0);
        let b = s.broadcast_to(&Shape::new(&[2, 3])).unwrap();
        assert_eq!(b.data(), &[3.0; 6]);
    }

    #[test]
    fn broadcast_row_to_matrix() {
        let row = Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]);
        let b = row.broadcast_to(&Shape::new(&[2, 3])).unwrap();
        assert_eq!(b.data(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn sum_axes_reduce_row() {
        let t = Tensor::from_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = t.sum_axes(&[0]);
        assert_eq!(s.data(), &[4.0, 6.0]);
        assert_eq!(s.shape().dims(), &[2]);
    }

    #[test]
    fn sum_total() {
        let t = Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(t.sum().data(), &[6.0]);
    }
}
