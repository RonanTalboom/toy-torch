//! Eager training of a 1D linear model:
//!
//!     y = sum(x * w) + b
//!
//! Pure autograd through the Tape. No graph compilation here — see
//! `examples/compile.rs` for the compiler pipeline.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use toy_torch::{Tape, Tensor, TensorId};

fn forward(tape: &mut Tape, x: TensorId, w: TensorId, b: TensorId) -> TensorId {
    let xw = tape.mul(x, w);
    let xw_sum = tape.sum(xw);
    tape.add(xw_sum, b)
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 50usize;
    let true_w = [2.5f32, -1.3, 0.8];
    let true_b = 0.4f32;

    // Generate a synthetic dataset.
    let mut samples: Vec<(Vec<f32>, f32)> = Vec::with_capacity(n);
    for _ in 0..n {
        let x: Vec<f32> = (0..3).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let y: f32 = x
            .iter()
            .zip(true_w.iter())
            .map(|(xi, wi)| xi * wi)
            .sum::<f32>()
            + true_b;
        samples.push((x, y));
    }

    // Parameters outside the tape so we can update them across iterations.
    let mut w_val = Tensor::from_vec(&[0.0, 0.0, 0.0], &[3]);
    let mut b_val = Tensor::scalar(0.0);
    let lr = 0.05f32;

    for epoch in 0..200 {
        let mut total_loss = 0.0f32;
        let mut grad_w = vec![0.0f32; 3];
        let mut grad_b = 0.0f32;

        for (x_vec, y_val) in &samples {
            let mut tape = Tape::new();
            let x = tape.frozen(Tensor::from_vec(x_vec, &[3]));
            let y_target = tape.frozen(Tensor::scalar(*y_val));
            let w = tape.leaf(w_val.clone());
            let b = tape.leaf(b_val.clone());

            let y_pred = forward(&mut tape, x, w, b);
            let diff = tape.sub(y_pred, y_target);
            let sq = tape.mul(diff, diff);
            let loss = tape.sum(sq);

            total_loss += tape.get(loss).data()[0];

            let grads = tape.backward(loss);
            let gw = grads.get(&w).expect("dL/dw");
            let gb = grads.get(&b).expect("dL/db");
            for (acc, v) in grad_w.iter_mut().zip(gw.data().iter()) {
                *acc += *v;
            }
            grad_b += gb.data()[0];
        }

        // Vanilla SGD step (average-grad, scaled by lr).
        for (wi, gi) in w_val.data_mut().iter_mut().zip(grad_w.iter()) {
            *wi -= lr * gi / n as f32;
        }
        b_val.data_mut()[0] -= lr * grad_b / n as f32;

        if epoch % 20 == 0 || epoch == 199 {
            println!(
                "epoch {epoch:>3}  loss={loss:.6}  w={w:?}  b={b:.4}",
                loss = total_loss / n as f32,
                w = w_val.data(),
                b = b_val.data()[0]
            );
        }
    }

    println!("target w={:?}  b={}", true_w, true_b);
    println!("learnt w={:?}  b={}", w_val.data(), b_val.data()[0]);
}
