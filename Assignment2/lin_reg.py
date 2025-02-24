import time
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import TypedDict, Callable, List, Dict
from enum import Enum
from copy import deepcopy

tf.compat.v1.enable_eager_execution()

# Feature flags
USE_PATIENCE_SCHEDULING = True
USE_SEED = True
USE_NOISE_SCALE = True
USE_NOISE_IN_DATA = True
USE_NOISE_IN_WEIGHTS = True
USE_NOISE_IN_LR = True

# Set seed for reproducibility
AUTHOR = "morgan"
SEED = sum(ord(char) for char in AUTHOR)
if USE_SEED:
    tf.random.set_seed(SEED)

# Create data
NUM_EXAMPLES = 10_000

# Noise type enum
class NoiseType(str, Enum):
    GAUSSIAN = 'Gaussian'
    UNIFORM = 'Uniform'
    LAPLACE = 'Laplace'

NOISE_TYPE = NoiseType.GAUSSIAN
NOISE_SCALE_X = 0.5
NOISE_SCALE_Y = 2.0
NOISE_SCALE_WEIGHTS = 0.01
NOISE_SCALE_LR = 0.0001  # Noise scale for learning rate (small to avoid instability)

if NOISE_TYPE == NoiseType.GAUSSIAN:
    NOISE_X = tf.random.normal([NUM_EXAMPLES])
    NOISE_Y = tf.random.normal([NUM_EXAMPLES])
elif NOISE_TYPE == NoiseType.UNIFORM:
    NOISE_X = tf.random.uniform([NUM_EXAMPLES], minval=-1, maxval=1)
    NOISE_Y = tf.random.uniform([NUM_EXAMPLES], minval=-1, maxval=1)
elif NOISE_TYPE == NoiseType.LAPLACE:
    u_x = tf.random.uniform([NUM_EXAMPLES], minval=-0.5, maxval=0.5)
    NOISE_X = -1 * tf.sign(u_x) * tf.math.log(1 - 2 * tf.abs(u_x))
    u_y = tf.random.uniform([NUM_EXAMPLES], minval=-0.5, maxval=0.5)
    NOISE_Y = -1 * tf.sign(u_y) * tf.math.log(1 - 2 * tf.abs(u_y))

if USE_NOISE_SCALE:
    NOISE_X *= NOISE_SCALE_X
    NOISE_Y *= NOISE_SCALE_Y

X_CLEAN = tf.random.normal([NUM_EXAMPLES])
X = deepcopy(X_CLEAN)
if USE_NOISE_IN_DATA:
    X += NOISE_X
Y = deepcopy(X_CLEAN) * 3 + 2 + NOISE_Y

TRAIN_STEPS = 5000 #1000
INITIAL_LEARNING_RATE = 0.005 #0.001
PATIENCE = 50
INITIAL_W = tf.Variable(0.5) #tf.Variable(0.0)
INITIAL_B = tf.Variable(0.3) #tf.Variable(0.0)

class DeviceType(str, Enum):
    CPU = "/CPU:0"
    GPU = "/GPU:0"

DEVICE = DeviceType.GPU

# Define the linear predictor
def prediction(x, W, b):
    return W * x + b

# Define loss functions
def squared_loss(y, y_predicted):
    return tf.reduce_mean(tf.square(y - y_predicted))

def huber_loss(y, y_predicted, m=1.0):
    residual = y - y_predicted
    condition = tf.abs(residual) <= m
    squared = 0.5 * tf.square(residual)
    linear = m * tf.abs(residual) - 0.5 * m**2
    return tf.reduce_mean(tf.where(condition, squared, linear))

def absolute_loss(y, y_predicted):
    return tf.reduce_mean(tf.abs(y - y_predicted))

def hybrid_loss(y, y_predicted, alpha=0.5):
    l1_component = absolute_loss(y, y_predicted)
    l2_component = squared_loss(y, y_predicted)
    return alpha * l1_component + (1 - alpha) * l2_component

class LossData(TypedDict):
    fn: Callable
    W: tf.Variable
    b: tf.Variable
    losses: List
    graph_color: str
    learning_rate: float
    learing_rate_historic: List[float]
    best_loss: float
    patience_counter: int

    @classmethod
    def from_fn(cls, fn, graph_color) -> "LossData":
        return cls(
            {
                "fn": fn,
                "W": deepcopy(INITIAL_W),
                "b": deepcopy(INITIAL_B),
                "losses": [],
                "graph_color": graph_color,
                "learning_rate": deepcopy(INITIAL_LEARNING_RATE),
                "learing_rate_historic": [],
                "best_loss": float('inf'),
                "patience_counter": 0
            }
        )

class LossFn(str, Enum):
    Squared = 'Squared'
    Huber = 'Huber'
    Hybrid = "Hybrid"

# Dictionary to select loss function
LOSS_FUNCTIONS: Dict[LossFn, LossData] = {
    'Squared': LossData.from_fn(squared_loss, "r"),
    'Huber': LossData.from_fn(huber_loss, "g"),
    'Hybrid': LossData.from_fn(hybrid_loss, "b"),
}

TIME_ELAPSED: Dict[LossFn, List[float]] = {
    'Squared': [],
    'Huber': [],
    'Hybrid': []
}

# Function to add noise to weights and biases (called inside tape)
def add_weight_noise(W, b, noise_type, scale):
    if noise_type == NoiseType.GAUSSIAN:
        W_noise = tf.random.normal([], stddev=scale)
        b_noise = tf.random.normal([], stddev=scale)
    elif noise_type == NoiseType.UNIFORM:
        W_noise = tf.random.uniform([], minval=-scale, maxval=scale)
        b_noise = tf.random.uniform([], minval=-scale, maxval=scale)
    elif noise_type == NoiseType.LAPLACE:
        u_W = tf.random.uniform([], minval=-0.5, maxval=0.5)
        W_noise = -scale * tf.sign(u_W) * tf.math.log(1 - 2 * tf.abs(u_W))
        u_b = tf.random.uniform([], minval=-0.5, maxval=0.5)
        b_noise = -scale * tf.sign(u_b) * tf.math.log(1 - 2 * tf.abs(u_b))
    return W + W_noise, b + b_noise

# Function to add noise to learning rate
def add_lr_noise(lr, noise_type, scale):
    if noise_type == NoiseType.GAUSSIAN:
        lr_noise = tf.random.normal([], stddev=scale)
    elif noise_type == NoiseType.UNIFORM:
        lr_noise = tf.random.uniform([], minval=-scale, maxval=scale)
    elif noise_type == NoiseType.LAPLACE:
        u_lr = tf.random.uniform([], minval=-0.5, maxval=0.5)
        lr_noise = -scale * tf.sign(u_lr) * tf.math.log(1 - 2 * tf.abs(u_lr))
    noisy_lr = lr + lr_noise
    # Ensure learning rate doesn't go negative
    return tf.maximum(noisy_lr, 1e-6)

# Training loop
with tf.device(DEVICE.value):
  for step in range(TRAIN_STEPS):
      for current_loss in LOSS_FUNCTIONS.keys():
          start_time = time.time()
          W = LOSS_FUNCTIONS[current_loss]['W']
          b = LOSS_FUNCTIONS[current_loss]['b']
          losses = LOSS_FUNCTIONS[current_loss]['losses']
          loss_fn = LOSS_FUNCTIONS[current_loss]['fn']
          current_lr = LOSS_FUNCTIONS[current_loss]['learning_rate']
          lr_history = LOSS_FUNCTIONS[current_loss]['learing_rate_historic']
          
          if USE_PATIENCE_SCHEDULING:
              best_loss = LOSS_FUNCTIONS[current_loss]['best_loss']
              patience_counter = LOSS_FUNCTIONS[current_loss]['patience_counter']

          # Calculate gradients
          with tf.GradientTape() as tape:
              # Add noise to weights if enabled
              if USE_NOISE_IN_WEIGHTS:
                  W_noisy, b_noisy = add_weight_noise(W, b, NOISE_TYPE, NOISE_SCALE_WEIGHTS)
              else:
                  W_noisy, b_noisy = W, b
              y_pred = prediction(X, W_noisy, b_noisy)
              loss = loss_fn(Y, y_pred)
          gradients = tape.gradient(loss, [W, b])
          
          # Check if gradients are None
          if gradients[0] is None or gradients[1] is None:
              print(f"[{current_loss}] Step {step}: Warning - Gradients are None!")
              continue
          
          # Add noise to learning rate if enabled
          if USE_NOISE_IN_LR:
              effective_lr = add_lr_noise(current_lr, NOISE_TYPE, NOISE_SCALE_LR)
          else:
              effective_lr = current_lr

          # Update parameters
          W.assign_sub(effective_lr * gradients[0])
          b.assign_sub(effective_lr * gradients[1])
          current_loss_value = loss.numpy()
          losses.append(current_loss_value)

          # Apply patience scheduling if feature flag is enabled
          if USE_PATIENCE_SCHEDULING:
              if current_loss_value < best_loss:
                  LOSS_FUNCTIONS[current_loss]['best_loss'] = current_loss_value
                  LOSS_FUNCTIONS[current_loss]['patience_counter'] = 0
              else:
                  LOSS_FUNCTIONS[current_loss]['patience_counter'] = patience_counter + 1

              if patience_counter >= PATIENCE:
                  new_lr = current_lr / 2
                  LOSS_FUNCTIONS[current_loss]['learning_rate'] = new_lr
                  LOSS_FUNCTIONS[current_loss]['patience_counter'] = 0
                  if step % 100 == 0:
                      print(f"[{current_loss}] Step {step}: Learning rate reduced to {new_lr:.6f}")

          LOSS_FUNCTIONS[current_loss]['learing_rate_historic'].append(LOSS_FUNCTIONS[current_loss]['learning_rate'])
          TIME_ELAPSED[current_loss].append(time.time() - start_time)

          # Print progress
          if step % 100 == 0:
              print(f"[{current_loss}] Step {step}: Loss = {current_loss_value:.4f}, "
                    f"W = {W.numpy():.4f}, b = {b.numpy():.4f}, LR = {effective_lr:.6f}")
              
# Print final parameters
print("\nFinal Parameters:")
for current_loss in LOSS_FUNCTIONS:
    print(f"{current_loss} Loss: W = {LOSS_FUNCTIONS[current_loss]['W'].numpy():.4f}, "
          f"b = {LOSS_FUNCTIONS[current_loss]['b'].numpy():.4f}, "
          f"Loss = {LOSS_FUNCTIONS[current_loss]['losses'][-1]:.4f}, "
          f"Final LR = {LOSS_FUNCTIONS[current_loss]['learning_rate']:.6f}")

print(f"\nAverage Time per Step on {DEVICE} (seconds):")
for current_loss in LOSS_FUNCTIONS:
    avg_time = sum(TIME_ELAPSED[current_loss]) / TRAIN_STEPS
    print(f"{current_loss} Loss: {avg_time:.6f}")
print("\nDevice used:", tf.test.gpu_device_name() if DEVICE == '/GPU:0' else "CPU")

# Plotting results
plt.figure(figsize=(12, 5))

# Plot 1: Data points and fitted lines
plt.subplot(1, 3, 1)
plt.scatter(X.numpy(), Y.numpy(), c='cyan', label='Original Data', alpha=0.5)
for current_loss in LOSS_FUNCTIONS:
    plt.plot(X.numpy(), prediction(X, LOSS_FUNCTIONS[current_loss]['W'], 
            LOSS_FUNCTIONS[current_loss]['b']).numpy(), 
            LOSS_FUNCTIONS[current_loss]['graph_color'], 
            label=f'{current_loss} Loss Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Fitted Lines')
plt.legend()

# Plot 2: Loss curves
plt.subplot(1, 3, 2)
for current_loss in LOSS_FUNCTIONS:
    plt.plot(LOSS_FUNCTIONS[current_loss]['losses'], 
             LOSS_FUNCTIONS[current_loss]['graph_color'], 
             label=f'{current_loss} Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Loss Over Time')
plt.legend()

plt.subplot(1, 3, 3)
for current_loss in LOSS_FUNCTIONS:
    plt.plot(LOSS_FUNCTIONS[current_loss]['learing_rate_historic'], 
             LOSS_FUNCTIONS[current_loss]['graph_color'], 
             label=f'{current_loss} Lr')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Time')
plt.legend()

plt.tight_layout()
plt.show()