use toy_torch::tensor::allclose;
use toy_torch::{Tape, Tensor};

/// Finite-difference gradient check for a scalar loss wrt one parameter.
fn grad_check<F: Fn(&mut Tape, toy_torch::TensorId) -> toy_torch::TensorId>(
    init: Tensor,
    f: F,
    tol: f32,
) {
    // Analytic grad via backward.
    let mut tape = Tape::new();
    let p = tape.leaf(init.clone());
    let loss = f(&mut tape, p);
    let grads = tape.backward(loss);
    let ana = grads.get(&p).expect("grad for param").clone();

    // Numerical grad via finite differences.
    let h = 1e-3f32;
    let mut num = Tensor::zeros(init.shape().dims());
    for i in 0..init.numel() {
        let mut plus = init.clone();
        plus.data_mut()[i] += h;
        let mut t1 = Tape::new();
        let p1 = t1.leaf(plus);
        let l1 = f(&mut t1, p1);
        let v1 = t1.get(l1).data()[0];

        let mut minus = init.clone();
        minus.data_mut()[i] -= h;
        let mut t2 = Tape::new();
        let p2 = t2.leaf(minus);
        let l2 = f(&mut t2, p2);
        let v2 = t2.get(l2).data()[0];

        num.data_mut()[i] = (v1 - v2) / (2.0 * h);
    }

    assert!(
        allclose(&ana, &num, tol),
        "grad mismatch:\nanalytic={:?}\nnumerical={:?}",
        ana.data(),
        num.data()
    );
}

#[test]
fn add_self_grad_is_one() {
    // L = sum(x); dL/dx_i = 1
    let mut tape = Tape::new();
    let x = tape.leaf(Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]));
    let loss = tape.sum(x);
    let grads = tape.backward(loss);
    let g = grads.get(&x).unwrap();
    assert_eq!(g.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn mul_const_grad() {
    // L = sum(3 * x); dL/dx_i = 3
    let mut tape = Tape::new();
    let three = tape.constant(Tensor::scalar(3.0));
    let x = tape.leaf(Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]));
    let y = tape.mul(three, x);
    let loss = tape.sum(y);
    let grads = tape.backward(loss);
    let g = grads.get(&x).unwrap();
    assert_eq!(g.data(), &[3.0, 3.0, 3.0]);
}

#[test]
fn quadratic_grad_check() {
    // L = sum((x - 1)^2); dL/dx = 2*(x - 1)
    grad_check(
        Tensor::from_vec(&[0.5, 1.5, 2.0, -1.0], &[4]),
        |tape, x| {
            let one = tape.constant(Tensor::scalar(1.0));
            let diff = tape.sub(x, one);
            let sq = tape.mul(diff, diff);
            tape.sum(sq)
        },
        1e-2,
    );
}

#[test]
fn relu_grad_check() {
    grad_check(
        Tensor::from_vec(&[-1.0, -0.5, 0.5, 1.0, 2.0], &[5]),
        |tape, x| {
            let r = tape.relu(x);
            tape.sum(r)
        },
        1e-3,
    );
}

#[test]
fn neg_grad_check() {
    grad_check(
        Tensor::from_vec(&[1.0, 2.0, 3.0], &[3]),
        |tape, x| {
            let n = tape.neg(x);
            tape.sum(n)
        },
        1e-3,
    );
}
