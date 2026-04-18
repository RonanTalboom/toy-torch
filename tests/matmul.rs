use toy_torch::tensor::allclose;
use toy_torch::{Tape, Tensor};

#[test]
fn matmul_forward_matches_hand_computed() {
    let mut tape = Tape::new();
    let a = tape.leaf(Tensor::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let b = tape.leaf(Tensor::from_vec(
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[3, 2],
    ));
    let y = tape.matmul(a, b);
    let out = tape.get(y);
    assert_eq!(out.shape().dims(), &[2, 2]);
    assert_eq!(out.data(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_backward_vs_finite_diff() {
    // L = sum(A @ B); compare analytic gradient vs finite differences.
    let a_init = Tensor::from_vec(&[0.5, 1.0, -0.3, 2.0], &[2, 2]);
    let b_init = Tensor::from_vec(&[1.5, -0.5, 0.25, 0.7], &[2, 2]);

    let mut tape = Tape::new();
    let a = tape.leaf(a_init.clone());
    let b = tape.leaf(b_init.clone());
    let y = tape.matmul(a, b);
    let loss = tape.sum(y);
    let grads = tape.backward(loss);
    let ga = grads.get(&a).unwrap().clone();
    let gb = grads.get(&b).unwrap().clone();

    // Analytic: dL/dA = ones @ B^T  and  dL/dB = A^T @ ones, with ones being [2,2] filled.
    // Via finite differences on A:
    let h = 1e-3f32;
    let mut num_a = Tensor::zeros(&[2, 2]);
    for i in 0..4 {
        let mut plus = a_init.clone();
        plus.data_mut()[i] += h;
        let mut t1 = Tape::new();
        let ap = t1.leaf(plus);
        let bp = t1.leaf(b_init.clone());
        let yp = t1.matmul(ap, bp);
        let lp = t1.sum(yp);
        let vp = t1.get(lp).data()[0];

        let mut minus = a_init.clone();
        minus.data_mut()[i] -= h;
        let mut t2 = Tape::new();
        let am = t2.leaf(minus);
        let bm = t2.leaf(b_init.clone());
        let ym = t2.matmul(am, bm);
        let lm = t2.sum(ym);
        let vm = t2.get(lm).data()[0];

        num_a.data_mut()[i] = (vp - vm) / (2.0 * h);
    }
    assert!(
        allclose(&ga, &num_a, 1e-2),
        "dL/dA mismatch:\n analytic={:?}\n numeric ={:?}",
        ga.data(),
        num_a.data()
    );

    // And dL/dB via finite differences.
    let mut num_b = Tensor::zeros(&[2, 2]);
    for i in 0..4 {
        let mut plus = b_init.clone();
        plus.data_mut()[i] += h;
        let mut t1 = Tape::new();
        let ap = t1.leaf(a_init.clone());
        let bp = t1.leaf(plus);
        let yp = t1.matmul(ap, bp);
        let lp = t1.sum(yp);
        let vp = t1.get(lp).data()[0];

        let mut minus = b_init.clone();
        minus.data_mut()[i] -= h;
        let mut t2 = Tape::new();
        let am = t2.leaf(a_init.clone());
        let bm = t2.leaf(minus);
        let ym = t2.matmul(am, bm);
        let lm = t2.sum(ym);
        let vm = t2.get(lm).data()[0];

        num_b.data_mut()[i] = (vp - vm) / (2.0 * h);
    }
    assert!(
        allclose(&gb, &num_b, 1e-2),
        "dL/dB mismatch:\n analytic={:?}\n numeric ={:?}",
        gb.data(),
        num_b.data()
    );
}
