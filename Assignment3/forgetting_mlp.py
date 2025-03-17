# -*- coding: utf-8 -*-
"""
Author:-mcn97
Analyzing Forgetting in neural networks
"""
import os
import psutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses warnings, only shows errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Turn off oneDNN

def log_memory(step):
    print(f"{step}: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    
log_memory('BEFORE IMPORTS')
from enum import Enum
from typing import Annotated, TypedDict

import matplotlib.pyplot as plt
import numpy as np
log_memory('BEFORE TF')
import tensorflow as tf
log_memory('AFTER TF')
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from sklearn.model_selection import train_test_split
from tqdm import tqdm

log_memory('AFTER IMPORTS')


log_memory('START')

SEED = 163537897  # Computed from poly_hash('mcn97')
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Define MLP model using Keras Sequential API
def create_mlp(
    depth,
    hidden_units=256,
    regularizer=None,
    dropout: bool = False,
    dropout_rate: int = 0.2,
):
    assert depth > 0, "Depth must be greater than 0"

    hidden_layers = []
    for _ in range(depth):
        hidden_layers.extend(
            [
                tf.keras.layers.Dense(
                    hidden_units, activation="relu", kernel_regularizer=regularizer
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        if dropout:
            hidden_layers.append(tf.keras.layers.Dropout(rate=dropout_rate))

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(784,)),
            *hidden_layers,
            tf.keras.layers.Dense(
                10, activation="softmax", kernel_regularizer=regularizer
            ),
        ]
    )


class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class ExperimentParams(TypedDict):
    num_tasks_to_run: int
    num_epochs_per_task: int
    num_epochs_initial: int
    optimizer: Optimizer
    dropout: bool
    depth: int
    regularizer_name: str
    minibatch_size: int
    output_folder: str

    @classmethod
    def new(cls, *, depth, regularizer_name, optimizer, dropout, output_folder, **kwargs):
        return cls(
            {
                "num_tasks_to_run": 10,
                "minibatch_size": 32,
                "num_epochs_initial": 50,
                "num_epochs_per_task": 20,
                "regularizer_name": regularizer_name,
                "optimizer": optimizer,
                "depth": depth,
                "dropout": dropout,
                **kwargs,
            }
        )


def get_optimizer(params: ExperimentParams):
    opt_type = params["optimizer"]
    if opt_type == Optimizer.ADAM:
        initial_lr = 0.01
    elif opt_type == Optimizer.SGD:
        initial_lr = 0.1
    elif opt_type == Optimizer.RMSPROP:
        initial_lr = 0.01

    samples_per_task = 60000  # Fixed for MNIST
    total_epochs = params["num_epochs_initial"] + params["num_epochs_per_task"] * (
        params["num_tasks_to_run"] - 1
    )
    total_steps = (total_epochs * samples_per_task) // params["minibatch_size"]

    # Dynamic decay_steps: ~10% of total steps
    decay_steps = total_steps // 10  # Decay 10 times over training
    num_decays = total_steps / decay_steps
    decay_rate = 0.1 ** (1 / num_decays)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr, decay_steps=decay_steps, decay_rate=decay_rate
    )

    match opt_type:
        case Optimizer.ADAM:
            return tf.keras.optimizers.Adam(
                learning_rate=lr_schedule, clipnorm=1.0
            )  # Gradient clipping
        case Optimizer.SGD:
            return tf.keras.optimizers.SGD(learning_rate=lr_schedule, clipnorm=1.0)
        case Optimizer.RMSPROP:
            return tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, clipnorm=1.0)


# Function to permute dataset according to task's permutation
def permute_dataset(x, y, permutation):
    return x[:, permutation], y


# Training Function
def train(model, optimizer, x, y, epochs, minibatch_size, pbar_desc):
    # Early stopping variables
    best_val_loss = float("inf")
    patience = 5
    val_split = 0.2
    wait = 0
    best_weights = None

    # Split into training and validation sets
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x, y, test_size=val_split, random_state=SEED
    )

    # Create dataset for training
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split))
        .shuffle(buffer_size=10000)  # Same as your original
        .batch(minibatch_size)
        .prefetch(tf.data.AUTOTUNE)  # Optimize performance
    )

    pbar_epoch = tqdm(
        range(epochs), desc=f"Training - {pbar_desc}", leave=False, unit=" epoch"
    )
    for epoch in pbar_epoch:
        for batch_x, batch_y in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_x, training=True)
                loss = tf.keras.losses.categorical_crossentropy(batch_y, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        val_predictions = model(x_val, training=False)
        val_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_val, val_predictions)
        )
        pbar_epoch.write(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_weights = model.get_weights()  # Save best weights
        else:
            wait += 1
            if wait >= patience:
                pbar_epoch.write("Early stopping triggered")
                model.set_weights(best_weights)  # Restore best weights
                break

    return model


# Testing Function
def test(model, x, y):
    predictions = model(x, training=False)
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y, predictions))
    return accuracy.numpy()

regularizer_dict = {
    "NLL": None,  # No regularization
    "L1": tf.keras.regularizers.l1(0.01),  # L1 regularization
    "L2": tf.keras.regularizers.l2(0.01),  # L2 regularization
    "L1+L2": tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),  # Combined L1 and L2
}

log_memory('DATASET')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0  # Flatten and normalize
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0  # Flatten and normalize
y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encode labels
y_test = tf.keras.utils.to_categorical(y_test, 10)
log_memory('POST-DATASET')


def plot_forgetting(R, params: ExperimentParams):
    num_tasks = R.shape[0]  # Should be 10 in this case
    plt.figure(figsize=(10, 6))  # Set figure size for clarity

    for i in range(num_tasks):
        # Extract accuracies for task i after training on task i and all subsequent tasks
        accuracies = R[i:, i]
        # X-axis: tasks trained on, from task i to task 9
        tasks_trained = range(i, num_tasks)
        # Plot the accuracy trend for task i
        plt.plot(tasks_trained, accuracies, marker="o", label=f"Task {i}")

    plt.xlabel("Tasks Trained On (Task Number)")
    plt.ylabel("Accuracy on Task")
    plt.title("Decrease in Model Prediction Accuracy Over Tasks (Forgetting)")
    plt.legend(title="Task Evaluated", loc="best")  # Add legend to distinguish tasks
    plt.grid(True, linestyle="--", alpha=0.7)  # Add grid for better readability
    plt.savefig(
        f"{params['output_folder']}/forgetting_mlp_depth{params['depth']}_reg{params['regularizer_name']}_dropout{params['dropout']}.png"
    )
    plt.close()


def run_experiment(params: ExperimentParams):
    log_memory('STARTING-EXP')
    task_permutation = [
        np.random.permutation(784) for _ in range(params["num_tasks_to_run"])
    ]
    regularizer = regularizer_dict[params["regularizer_name"]]
    optimizer = get_optimizer(params)
    model = create_mlp(
        params["depth"], regularizer=regularizer, dropout=params["dropout"]
    )
    R = np.zeros(
        (params["num_tasks_to_run"], params["num_tasks_to_run"])
    )  # Performance matrix

    # Train on Task A (task 0)
    permutation = task_permutation[0]
    x_train_permuted, y_train_permuted = permute_dataset(x_train, y_train, permutation)
    x_test_permuted, y_test_permuted = permute_dataset(x_test, y_test, permutation)
    
    log_memory('BEFORE-EXP-TRAIN-A')
    model = train(
        model,
        optimizer,
        x_train_permuted,
        y_train_permuted,
        params["num_epochs_initial"],  # 50 epochs for Task A
        params["minibatch_size"],
        "Task A",
    )
    log_memory('AFTER-EXP-TRAIN-A')
    print("Testing Task A")
    R[0, 0] = test(model, x_test_permuted, y_test_permuted)

    # Train on subsequent tasks
    for t in tqdm(
        range(1, params["num_tasks_to_run"]),
        desc=f"SubTasks Training",
        leave=True,
    ):
        permutation = task_permutation[t]
        x_train_permuted, y_train_permuted = permute_dataset(
            x_train, y_train, permutation
        )
        x_test_permuted, y_test_permuted = permute_dataset(x_test, y_test, permutation)
        model = train(
            model,
            optimizer,
            x_train_permuted,
            y_train_permuted,
            params["num_epochs_per_task"],  # 20 epochs for subsequent tasks
            params["minibatch_size"],
            f"Subtask #{t}",
        )

        # Test on all tasks seen so far
        for i in tqdm(
            range(t + 1),
            desc=f"Testing all tasks (Subtask #{t})",
            leave=False,
            position=1,
        ):
            permutation_i = task_permutation[i]
            x_test_permuted_i, y_test_permuted_i = permute_dataset(
                x_test, y_test, permutation_i
            )
            R[t, i] = test(model, x_test_permuted_i, y_test_permuted_i)

    # Compute metrics
    T = params["num_tasks_to_run"]
    ACC = np.mean(R[T - 1, :])  # Average accuracy after training on all tasks

    bwt_data = [R[T - 1, i] - R[i, i] for i in range(T - 1)]
    BWT = np.mean(bwt_data)  # Backward transfer
    TBWT = np.sum(bwt_data)  # Total Backward Transfer
    CBWT = np.mean([R[i, i] - R[T - 1, i] for i in range(T - 1)])  # Average Forgetting

    print(f"Params: {params}")
    print(f'Regularizer: {params["regularizer_name"]}')
    print(f'Depth: {params["depth"]}')
    print(f'With dropout: {params["dropout"]}')
    print(f"ACC: {ACC:.4f}")
    print(f"BWT: {BWT:.4f}")
    print(f"TBWT: {TBWT:.4f}")
    print(f"CBWT: {CBWT:.4f}")
    plot_forgetting(R, params)


class Settings(BaseSettings):
    depth: Annotated[int, Field(ge=1)] = 2
    optimizer: Optimizer = Optimizer.ADAM
    regularizer: str = list(regularizer_dict.keys())[0]
    dropout: bool = True
    output_folder: str = "./"

    @field_validator("regularizer")
    def validate_regilarizer(value: str) -> str:
        if value not in regularizer_dict:
            raise ValueError(f"Invalid regularizer, must be: {regularizer_dict.keys()}")
        return value


if __name__ == "__main__":
    settings = Settings()
    params = ExperimentParams.new(
        depth=settings.depth,
        regularizer_name=settings.regularizer,
        optimizer=settings.optimizer,
        dropout=settings.dropout,
        output_folder=settings.output_folder,
        num_tasks_to_run=2,
        num_epochs_per_task=1,
        num_epochs_initial=1,
    )
    run_experiment(params)
