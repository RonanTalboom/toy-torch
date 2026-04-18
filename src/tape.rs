//! Arena-based autograd tape.
//!
//! Every tensor produced through the tape gets a stable [`TensorId`] and lives
//! in the arena. The tape records the [`Op`] and input IDs in order. Backward
//! walks the records in reverse to propagate gradients.
//!
//! Design choice (load-bearing for pedagogy): the tape is an **explicit `&mut`
//! argument** at every op site, not a thread-local or `Rc<RefCell>`. This
//! makes the dispatch pattern visible in the type system. See the vault note
//! [[Arena-based autograd tapes teach dispatch separation better than Rc-based tapes]].

use crate::op::Op;
use crate::tensor::Tensor;

/// Stable handle to a tensor on the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

#[derive(Debug, Clone)]
pub struct TapeNode {
    pub op: Op,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    /// Whether this node's inputs require gradients. Leaf tensors can set this
    /// to `false` to exclude themselves from backward (e.g., frozen data).
    pub requires_grad: bool,
}

#[derive(Debug, Default)]
pub struct Tape {
    /// Arena of realized tensor values (forward pass outputs).
    pub(crate) tensors: Vec<Tensor>,
    /// Ordered record of operations.
    pub(crate) nodes: Vec<TapeNode>,
    /// Which tensor IDs are leaves that require grad (trained parameters).
    pub(crate) requires_grad: Vec<bool>,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, id: TensorId) -> &Tensor {
        &self.tensors[id.0]
    }

    pub fn nodes(&self) -> &[TapeNode] {
        &self.nodes
    }

    pub(crate) fn push_tensor(&mut self, t: Tensor, requires_grad: bool) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(t);
        self.requires_grad.push(requires_grad);
        id
    }

    /// Add a leaf (trainable parameter or input). Sets `requires_grad = true`
    /// by default — flip to `false` for frozen inputs via [`Tape::frozen`].
    pub fn leaf(&mut self, t: Tensor) -> TensorId {
        let id = self.push_tensor(t, true);
        self.nodes.push(TapeNode {
            op: Op::Leaf,
            inputs: vec![],
            output: id,
            requires_grad: true,
        });
        id
    }

    /// Add a frozen (non-trainable) input. Its gradient will not be computed.
    pub fn frozen(&mut self, t: Tensor) -> TensorId {
        let id = self.push_tensor(t, false);
        self.nodes.push(TapeNode {
            op: Op::Leaf,
            inputs: vec![],
            output: id,
            requires_grad: false,
        });
        id
    }

    /// Add a compile-time constant. Distinct from `leaf` so constant-folding
    /// can collapse chains of `Const` without disturbing `Leaf` parameters.
    pub fn constant(&mut self, t: Tensor) -> TensorId {
        let id = self.push_tensor(t, false);
        self.nodes.push(TapeNode {
            op: Op::Const,
            inputs: vec![],
            output: id,
            requires_grad: false,
        });
        id
    }

    fn record(&mut self, op: Op, inputs: Vec<TensorId>, out: Tensor) -> TensorId {
        let rg = inputs.iter().any(|&i| self.requires_grad[i.0]);
        let id = self.push_tensor(out, rg);
        self.nodes.push(TapeNode {
            op,
            inputs,
            output: id,
            requires_grad: rg,
        });
        id
    }

    // ---- elementwise binary ----

    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = elementwise_binary(&self.tensors[a.0], &self.tensors[b.0], |x, y| x + y);
        self.record(Op::Add, vec![a, b], out)
    }

    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = elementwise_binary(&self.tensors[a.0], &self.tensors[b.0], |x, y| x - y);
        self.record(Op::Sub, vec![a, b], out)
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = elementwise_binary(&self.tensors[a.0], &self.tensors[b.0], |x, y| x * y);
        self.record(Op::Mul, vec![a, b], out)
    }

    // ---- elementwise unary ----

    pub fn neg(&mut self, a: TensorId) -> TensorId {
        let ta = &self.tensors[a.0];
        let data: Vec<f32> = ta.data().iter().map(|x| -x).collect();
        let out = Tensor::new(data, ta.shape().clone()).expect("neg shape ok");
        self.record(Op::Neg, vec![a], out)
    }

    pub fn relu(&mut self, a: TensorId) -> TensorId {
        let ta = &self.tensors[a.0];
        let data: Vec<f32> = ta.data().iter().map(|x| x.max(0.0)).collect();
        let out = Tensor::new(data, ta.shape().clone()).expect("relu shape ok");
        self.record(Op::Relu, vec![a], out)
    }

    // ---- matmul ----

    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = self.tensors[a.0]
            .matmul2d(&self.tensors[b.0])
            .expect("matmul: shape mismatch");
        self.record(Op::Matmul, vec![a, b], out)
    }

    // ---- reduce ----

    pub fn sum(&mut self, a: TensorId) -> TensorId {
        let out = self.tensors[a.0].sum();
        self.record(Op::Sum, vec![a], out)
    }
}

/// Broadcast-aware elementwise binary kernel. Works on contiguous backing arrays.
pub(crate) fn elementwise_binary<F: Fn(f32, f32) -> f32>(a: &Tensor, b: &Tensor, f: F) -> Tensor {
    let out_shape = crate::shape::Shape::broadcast(a.shape(), b.shape())
        .expect("elementwise_binary: shapes cannot broadcast");
    let ab = a.broadcast_to(&out_shape).expect("broadcast a");
    let bb = b.broadcast_to(&out_shape).expect("broadcast b");
    let data: Vec<f32> = ab
        .data()
        .iter()
        .zip(bb.data().iter())
        .map(|(x, y)| f(*x, *y))
        .collect();
    Tensor::new(data, out_shape).expect("elementwise_binary: internal")
}
