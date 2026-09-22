"""
Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_synthetic_regression_batch
def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    np.random.seed(seed)
    x = np.random.randn(batch_size,in_dim)
    w = np.random.randn(in_dim,out_dim)
    noise = 0.1*np.random.randn(batch_size,out_dim)
    y = x@w + noise
    return x.astype(np.float64), y.astype(np.float64)

# Step 2 - init_mlp_params
def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    np.random.seed(seed)

    std1 = np.sqrt(2/in_dim)
    W1 = np.random.randn(in_dim,hidden_dim) * std1
    b1 = np.zeros(hidden_dim, dtype=  np.float64)

    std2 = np.sqrt(2/hidden_dim)
    W2 = np.random.randn(hidden_dim,out_dim) * std2
    b2 = np.zeros(out_dim, dtype=  np.float64)

    return {
        "W1": W1.astype(np.float64),
        "b1": b1,
        "W2": W2.astype(np.float64),
        "b2": b2
    }

# Step 3 - linear_forward
def linear_forward(x, w, b):
    y = x @ w +b 
    return y

# Step 4 - relu_forward
def relu_forward(x):
    return np.maximum(x,0.0)

# Step 5 - mlp_forward
def mlp_forward(x, params):
    z1 = linear_forward(x,params["W1"], params["b1"])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1,params["W2"],params["b2"])
    cache = {
        "x" : x,
        "z1"  : z1,
        "a1"  :a1,
        "z2" : z2
    }
    return z2, cache

# Step 6 - mse_loss_and_grad
def mse_loss_and_grad(y_pred, y_true):
    diff = y_pred  - y_true
    loss = np.mean(diff**2)
    dy_pred = (2/y_pred.size) * diff
    return float(loss) , dy_pred

# Step 7 - linear_backward
import numpy as np

def linear_backward(d_out, x, w):
    dx = d_out @ w.T
    dw = x.T @ d_out
    db = np.sum(d_out, axis = 0)
    return dx, dw, db

# Step 8 - relu_backward
def relu_backward(d_out, z):
    dz = d_out * (z>0)
    return dz

# Step 9 - first_linear_backward
def first_linear_backward(d_z1, x, w1):
    dx, dW1, db1 = linear_backward(d_z1,x,w1)
    return dx,dW1,db1

# Step 10 - mlp_backward
def mlp_backward(dy_pred, cache, params):
    da1, dW2, db2 = linear_backward(dy_pred, cache["a1"],params["W2"])
    dz1 = relu_backward(da1,cache["z1"])
    _, dW1, db1 = first_linear_backward(dz1, cache["x"],params["W1"])
    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }

# Step 11 - split_into_micro_batches
def split_into_micro_batches(x, y, micro_batch_size):
    N = x.shape[0]
    batches = []
    for i in range(0,N,micro_batch_size):
        x_mb = x[i:i+micro_batch_size]
        y_mb = y[i:i+micro_batch_size]
        batches.append((x_mb,y_mb))
    return batches

# Step 12 - accumulate_gradients (not yet solved)
# TODO: implement

# Step 13 - scale_accumulated_gradients (not yet solved)
# TODO: implement

# Step 14 - grad_accumulation_step (not yet solved)
# TODO: implement

# Step 15 - mlp_forward_checkpointed (not yet solved)
# TODO: implement

# Step 16 - recompute_block_activations (not yet solved)
# TODO: implement

# Step 17 - mlp_backward_checkpointed (not yet solved)
# TODO: implement

# Step 18 - estimate_checkpointing_memory_savings (not yet solved)
# TODO: implement

# Step 19 - cast_to_half_precision (not yet solved)
# TODO: implement

# Step 20 - make_master_params (not yet solved)
# TODO: implement

# Step 21 - scale_loss (not yet solved)
# TODO: implement

# Step 22 - unscale_gradients (not yet solved)
# TODO: implement

# Step 23 - has_non_finite_gradients (not yet solved)
# TODO: implement

# Step 24 - mixed_precision_step (not yet solved)
# TODO: implement

# Step 25 - shard_dataset_across_workers (not yet solved)
# TODO: implement

# Step 26 - compute_local_gradients (not yet solved)
# TODO: implement

# Step 27 - all_reduce_mean (not yet solved)
# TODO: implement

# Step 28 - ring_all_reduce_mean (not yet solved)
# TODO: implement

# Step 29 - data_parallel_train_step (not yet solved)
# TODO: implement

# Step 30 - bucket_gradients (not yet solved)
# TODO: implement

# Step 31 - init_adam_state (not yet solved)
# TODO: implement

# Step 32 - partition_optimizer_state (not yet solved)
# TODO: implement

# Step 33 - local_shard_adam_update (not yet solved)
# TODO: implement

# Step 34 - all_gather_param_shards (not yet solved)
# TODO: implement

# Step 35 - zero_optimizer_step (not yet solved)
# TODO: implement

# Step 36 - compute_param_memory_bytes (not yet solved)
# TODO: implement

# Step 37 - compute_optimizer_memory_bytes (not yet solved)
# TODO: implement

# Step 38 - compute_peak_activation_memory_bytes (not yet solved)
# TODO: implement

# Step 39 - compare_memory_with_and_without_optimizations (not yet solved)
# TODO: implement

# Step 40 - full_distributed_training_loop (not yet solved)
# TODO: implement

