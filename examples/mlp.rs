//! A small MLP trained via SGD:
//!
//!     h = relu(x @ W1 + b1)
//!     y = h @ W2 + b2
//!     loss = sum((y - target)^2)
//!
//! Exercises matmul + broadcast-add + relu + sum all together. Intentionally
//! tiny so the training loop fits on one screen.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use toy_torch::{Tape, Tensor, TensorId};

const IN_DIM: usize = 4;
const HIDDEN: usize = 8;
const OUT_DIM: usize = 1;
const N_SAMPLES: usize = 64;

fn kaiming(rng: &mut StdRng, fan_in: usize, dims: &[usize]) -> Tensor {
    let n: usize = dims.iter().product();
    let scale = (2.0 / fan_in as f32).sqrt();
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0) * scale).collect();
    Tensor::from_vec(&data, dims)
}

fn forward(
    tape: &mut Tape,
    x: TensorId,
    w1: TensorId,
    b1: TensorId,
    w2: TensorId,
    b2: TensorId,
) -> TensorId {
    let xw1 = tape.matmul(x, w1);
    let pre = tape.add(xw1, b1);
    let h = tape.relu(pre);
    let hw2 = tape.matmul(h, w2);
    tape.add(hw2, b2)
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);

    // Synthetic target function: a simple dot product plus a bias.
    let true_w = [0.5f32, -1.0, 0.3, 2.0];
    let true_b = 0.25f32;

    // Build dataset as one big [N, 4] batch.
    let mut x_data = Vec::with_capacity(N_SAMPLES * IN_DIM);
    let mut y_data = Vec::with_capacity(N_SAMPLES);
    for _ in 0..N_SAMPLES {
        let row: Vec<f32> = (0..IN_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let y: f32 = row
            .iter()
            .zip(true_w.iter())
            .map(|(xi, wi)| xi * wi)
            .sum::<f32>()
            + true_b;
        y_data.push(y);
        x_data.extend(row);
    }
    let x_batch = Tensor::from_vec(&x_data, &[N_SAMPLES, IN_DIM]);
    let y_target = Tensor::from_vec(&y_data, &[N_SAMPLES, OUT_DIM]);

    // Initialize parameters with Kaiming-style random weights.
    let mut w1_val = kaiming(&mut rng, IN_DIM, &[IN_DIM, HIDDEN]);
    let mut b1_val = Tensor::zeros(&[HIDDEN]);
    let mut w2_val = kaiming(&mut rng, HIDDEN, &[HIDDEN, OUT_DIM]);
    let mut b2_val = Tensor::zeros(&[OUT_DIM]);

    let lr = 0.05f32;

    for epoch in 0..500 {
        let mut tape = Tape::new();
        let x = tape.frozen(x_batch.clone());
        let y_t = tape.frozen(y_target.clone());
        let w1 = tape.leaf(w1_val.clone());
        let b1 = tape.leaf(b1_val.clone());
        let w2 = tape.leaf(w2_val.clone());
        let b2 = tape.leaf(b2_val.clone());

        let y_pred = forward(&mut tape, x, w1, b1, w2, b2);
        let diff = tape.sub(y_pred, y_t);
        let sq = tape.mul(diff, diff);
        let loss = tape.sum(sq);

        let loss_val = tape.get(loss).data()[0] / N_SAMPLES as f32;

        let grads = tape.backward(loss);

        sgd_step(&mut w1_val, grads.get(&w1).unwrap(), lr, N_SAMPLES);
        sgd_step(&mut b1_val, grads.get(&b1).unwrap(), lr, N_SAMPLES);
        sgd_step(&mut w2_val, grads.get(&w2).unwrap(), lr, N_SAMPLES);
        sgd_step(&mut b2_val, grads.get(&b2).unwrap(), lr, N_SAMPLES);

        if epoch % 50 == 0 || epoch == 499 {
            println!("epoch {epoch:>3}  loss={loss_val:.6}");
        }
    }

    println!("done. final w1[0..4]={:?}", &w1_val.data()[..4]);
}

fn sgd_step(param: &mut Tensor, grad: &Tensor, lr: f32, batch_size: usize) {
    let k = lr / batch_size as f32;
    for (p, g) in param.data_mut().iter_mut().zip(grad.data().iter()) {
        *p -= k * g;
    }
}
