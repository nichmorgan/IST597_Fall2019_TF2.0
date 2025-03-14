# -*- coding: utf-8 -*-
"""
Author:-aam35
Analyzing Forgetting in neural networks
"""
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppresses warnings, only shows errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turn off oneDNN

# Define MLP model using Keras Sequential API
def create_mlp(depth, hidden_units=256, regularizer=None):
	assert depth > 0, 'Depth must be greater than 0'
	return tf.keras.Sequential([
		tf.keras.layers.Input(shape=(784,)),
		*([tf.keras.layers.Dense(hidden_units, activation='relu', kernel_regularizer=regularizer) for _ in range(depth)]),
		tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=regularizer)
	])

# Function to permute dataset according to task's permutation
def permute_dataset(x, y, permutation):
	return x[:, permutation], y

# Training Function
def train(model, x, y, epochs, minibatch_size, learning_rate):
	dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(minibatch_size)
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	for _ in tqdm(range(epochs), desc='Training', leave=False, unit=' epoch'):
		for batch_x, batch_y in dataset:
			with tf.GradientTape() as tape:
				predictions = model(batch_x, training=True)
				loss = tf.keras.losses.categorical_crossentropy(batch_y, predictions)
			gradients = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Testing Function
def test(model, x, y):
	predictions = model(x, training=False)
	accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y, predictions))
	return accuracy.numpy()

# Hyperparameters
minibatch_size = 32
learning_rate = 0.001
num_epochs_initial = 50 	# For Task A
num_epochs_per_task = 20 	# For subsequent tasks
depths = [2, 3, 4] 			# MLP depths to evaluate

# Define regularizers for different loss functions
regularizer_dict = {
	'NLL': None,											# No regularization
	'L1': tf.keras.regularizers.l1(0.01),					# L1 regularization
	'L2': tf.keras.regularizers.l2(0.01),					# L2 regularization
	'L1+L2': tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),	# Combined L1 and L2
}

class DeviceType(str, Enum):
    CPU = "/CPU:0"
    GPU = "/GPU:0"

DEVICE = DeviceType.GPU

# Run experiments for each depth
with tf.device(DEVICE.value):
	## Permuted MNIST
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.reshape(-1, 784).astype('float32') / 255.0 # Flatten and normalize
	x_test = x_test.reshape(-1, 784).astype('float32') / 255.0 # Flatten and normalize
	y_train = tf.keras.utils.to_categorical(y_train, 10) # One-hot encode labels
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	# Generate the tasks specifications as a list of random permutations of the input pixels.
	num_tasks_to_run = 10
	task_permutation = [np.random.permutation(784) for _ in range(num_tasks_to_run)]

	reg_pbar = tqdm(regularizer_dict.items(), leave=False)
	for reg_name, regularizer in reg_pbar:
		reg_pbar.set_description(f'Regularizer {reg_name}')

		experiments_pbar = tqdm(depths, desc='Experiments', leave=False)
		for depth in experiments_pbar:
			model = create_mlp(depth, regularizer=regularizer)
			R = np.zeros((num_tasks_to_run, num_tasks_to_run)) # Performance matrix
			
			# Train on Task A (task 0)
			permutation = task_permutation[0]
			x_train_permuted, y_train_permuted = permute_dataset(x_train, y_train, permutation)
			x_test_permuted, y_test_permuted = permute_dataset(x_test, y_test, permutation)
			train(model, x_train_permuted, y_train_permuted, num_epochs_initial, minibatch_size, learning_rate)
			R[0, 0] = test(model, x_test_permuted, y_test_permuted)
			
			# Train on subsequent tasks
			for t in tqdm(range(1, num_tasks_to_run), desc='SubTasks Training', leave=False):
				permutation = task_permutation[t]
				x_train_permuted, y_train_permuted = permute_dataset(x_train, y_train, permutation)
				x_test_permuted, y_test_permuted = permute_dataset(x_test, y_test, permutation)
				train(model, x_train_permuted, y_train_permuted, num_epochs_per_task, minibatch_size, learning_rate)
				
				# Test on all tasks seen so far
				for i in tqdm(range(t+1), desc='Testing all tasks', leave=False):
					permutation_i = task_permutation[i]
					x_test_permuted_i, y_test_permuted_i = permute_dataset(x_test, y_test, permutation_i)
					R[t, i] = test(model, x_test_permuted_i, y_test_permuted_i)
			
			# Compute metrics
			T = num_tasks_to_run
			ACC = np.mean(R[T-1, :]) # Average accuracy after training on all tasks
			BWT = np.mean(R[T-1, i] - R[i, i] for i in range(T-1)) # Backward transfer
			experiments_pbar.update()
			experiments_pbar.write(f'Regularizer: {reg_name}, Depth: {depth}, ACC: {ACC:.4f}, BWT: {BWT:.4f}')