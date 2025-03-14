from enum import Enum
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from utils.mnist_reader import load_mnist
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255.0)
    label = tf.cast(label, tf.int64)
    return image, label

# Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 100
img_shape = (28, 28)
train_split_n = 50000

class DeviceType(str, Enum):
    CPU = "/CPU:0"
    GPU = "/GPU:0"

DEVICE = DeviceType.CPU

class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    
model_type = ModelType.LOGISTIC_REGRESSION

# Step 1: Read in data
fmnist_folder = 'data/fashion'
with tqdm(total=2, desc='Reading data') as pbar:
    X_train_full, y_train_full = load_mnist(fmnist_folder, kind='train')
    pbar.update(1)
    X_test, y_test = load_mnist(fmnist_folder, kind='t10k')
    pbar.update(1)

# Create train/validation split
X_train = X_train_full[:train_split_n]
y_train = y_train_full[:train_split_n]
X_val = X_train_full[train_split_n:]
y_val = y_train_full[train_split_n:]

# Preprocess data for scikit-learn models (flatten and normalize)
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_val_flat = X_val.reshape(X_val.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

# Step 2: Create datasets
with tqdm(total=3, desc='Creating datasets') as pbar:
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(preprocess_image).shuffle(buffer_size=10000).batch(batch_size)
    pbar.update(1)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(preprocess_image).batch(batch_size)
    pbar.update(1)
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(preprocess_image).batch(batch_size)
    pbar.update(1)

# Step 3: Create weights and bias
w = tf.Variable(tf.random.normal([784, 10], mean=0.0, stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([10]), name='bias')

match model_type:
    case ModelType.LOGISTIC_REGRESSION:
        # Step 4: Build model
        def logits_fn(img):
            return tf.matmul(img, w) + b

        # Step 5: Define loss function
        def loss_fn(logits, label):
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))

        # Step 6: Define optimizer
        class GradientDescent(tf.Module):

            def __init__(self, learning_rate):
                # Initialize parameters
                self.learning_rate = learning_rate
                self.title = f"Gradient descent optimizer: learning rate={self.learning_rate}"

            def apply_gradients(self, grads_and_vars):
                # Update variables
                for grad, vars in grads_and_vars:
                    vars.assign_sub(self.learning_rate*grad)

        class Momentum(tf.Module):

            def __init__(self, learning_rate, momentum=0.7, variables=[w, b]):
                # Initialize parameters
                self.learning_rate = learning_rate
                self.momentum = momentum
                self.changes = [
                    tf.Variable(tf.zeros_like(var), name=f'change_{i}') 
                    for i, var in enumerate(variables)
                ]
                self.title = f"Gradient descent optimizer: learning rate={self.learning_rate}"

            def apply_gradients(self, grads_and_vars):
                # Update variables 
                for (grad, vars), change_var in zip(grads_and_vars, self.changes):
                    curr_change = self.learning_rate*grad + self.momentum*change_var
                    vars.assign_sub(curr_change)
                    change_var.assign(curr_change)

        optimizer = Momentum(learning_rate)

        # Step 7: Training loop with train/val accuracy reporting
        train_acc_history = []
        val_acc_history = []
        train_times = []
        with tf.device(DEVICE.value):
            with tqdm(total=n_epochs, desc="Training", unit=" epoch", leave=False) as pbar:
                for i in range(n_epochs):
                    start_time = time.time()
                    
                    # Training
                    total_loss = 0
                    n_batches = 0
                    train_correct = 0
                    train_total = 0
                    for img_batch, label_batch in train_data:
                        with tf.GradientTape() as tape:
                            logits = logits_fn(img_batch)
                            loss_value = loss_fn(logits, label_batch)
                        grads = tape.gradient(loss_value, [w, b])
                        optimizer.apply_gradients(zip(grads, [w, b]))
                        total_loss += loss_value.numpy()
                        n_batches += 1
                        
                        # Compute training accuracy
                        preds = tf.argmax(logits, axis=1)
                        correct = tf.reduce_sum(tf.cast(tf.equal(preds, label_batch), tf.float32)).numpy()
                        train_correct += correct
                        train_total += label_batch.shape[0]
                    
                    train_accuracy = train_correct / train_total
                    
                    # Validation
                    val_accuracy = 0
                    val_batches = 0
                    for img_batch, label_batch in val_data:
                        logits = logits_fn(img_batch)
                        preds = tf.argmax(logits, axis=1)
                        accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, label_batch), tf.float32)).numpy()
                        val_accuracy += accuracy
                        val_batches += 1
                    val_accuracy /= val_batches
                    
                    epoch_time = time.time() - start_time
                    train_acc_history.append(train_accuracy)
                    val_acc_history.append(val_accuracy)
                    train_times.append(epoch_time)
                    pbar.set_description(f"Loss: {total_loss/n_batches:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.2f}s")
                    pbar.update(1)

            # Step 8: Final test accuracy
            test_accuracy = 0
            test_batches = 0
            for img_batch, label_batch in tqdm(test_data, desc="Testing", unit=" batch"):
                logits = logits_fn(img_batch)
                preds = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, label_batch), tf.float32)).numpy()
                test_accuracy += accuracy
                test_batches += 1
            test_accuracy /= test_batches
            print(f"Final Test Accuracy: {test_accuracy:.4f}")

        def plot_train_times(train_times):
            avg_epoch_time = np.mean(train_times)
            plt.plot(range(1, n_epochs+1), train_times)
            plt.axhline(y=avg_epoch_time, color='r', linestyle='--', label=f'Average Time: {avg_epoch_time:.2f}s')
            plt.xlabel('Epoch')
            plt.ylabel('Time (s)')
            plt.title('Epoch Time Over Time')
            plt.legend()
            plt.savefig(f'./assets/q2/epoch_times_{n_epochs}_epochs_{DEVICE}.png')
            plt.close()

        plot_train_times(train_times)
                
        # Step 9: Plot images
        def plot_images(images, y, yhat=None):
            assert len(images) == len(y) == 9
            fig, axes = plt.subplots(3, 3)
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i].reshape(img_shape), cmap='binary')
                if yhat is None:
                    xlabel = f"True: {y[i]}"
                else:
                    xlabel = f"True: {y[i]}, Pred: {yhat[i]}"
                ax.set_xlabel(xlabel)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.savefig(f'./assets/q2/images_{n_epochs}_epochs.png')
            plt.close()

        images = X_test[:9].reshape(9, 28, 28)
        y = y_test[:9]
        plot_images(images=images, y=y)

        # Step 10: Plot weights
        def plot_weights(w):
            w_val = w.numpy()
            w_min = np.min(w_val)
            w_max = np.max(w_val)
            fig, axes = plt.subplots(3, 4)
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            for i, ax in enumerate(axes.flat):
                if i < 10:
                    image = w_val[:, i].reshape(img_shape)
                    ax.set_xlabel(f"Weights: {i}")
                    ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
                ax.set_xticks([])
                ax.set_yticks([])
            plt.savefig(f'./assets/q2/weights_{n_epochs}_epochs.png')
            plt.close()

        plot_weights(w)

        def plot_accuracies(n_epochs, train_acc_history, val_acc_history):
            # Plot Train/Val accuracy over time
            plt.plot(range(1, n_epochs+1), train_acc_history, label=f'Train Accuracy: {train_acc_history[-1]:.4f}')
            plt.plot(range(1, n_epochs+1), val_acc_history, label=f'Validation Accuracy: {val_acc_history[-1]:.4f}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Train and Validation Accuracy Over Time')
            plt.legend()
            plt.savefig(f'./assets/q2/accuracy_{n_epochs}_epochs.png')
            plt.close()
            
        plot_accuracies(n_epochs, train_acc_history, val_acc_history)
        
        def plot_clusters(w):
            w_val = w.numpy().T # Transpose to get shape (10, 784)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=3)
            w_2d = tsne.fit_transform(w_val)
            
            class_names = [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]
            
            # Plot clusters
            plt.figure(figsize=(8, 6))
            for i in range(10):
                plt.scatter(w_2d[i, 0], w_2d[i, 1], label=class_names[i])
            plt.legend(loc='best')
            plt.title('t-SNE Visualization of Class Weights')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.savefig(f'./assets/q2/tsne_{n_epochs}_epochs.png')
            plt.close()
        
        plot_clusters(w)
    
    case ModelType.SVM:
        svm = SVC(kernel='rbf', random_state=42)
        print("Training SVM...")
        svm.fit(X_train_flat, y_train)
        
        print("Evaluating SVM...")
        svm_val_pred = svm.predict(X_val_flat)
        svm_val_acc = accuracy_score(y_val, svm_val_pred)
        
        print("Testing SVM...")
        svm_test_pred = svm.predict(X_test_flat)
        svm_test_acc = accuracy_score(y_test, svm_test_pred)
        
        print(f'SVM - Validation Accuracy: {svm_val_acc:.4f}, Test Accuracy: {svm_test_acc:.4f}')

    case ModelType.RANDOM_FOREST:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Training Random Forest...")
        rf.fit(X_train_flat, y_train)
        
        print("Evaluating Random Forest...")
        rf_val_pred = rf.predict(X_val_flat)
        rf_val_acc = accuracy_score(y_val, rf_val_pred)
        
        print("Testing Random Forest...")
        rf_test_pred = rf.predict(X_test_flat)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)
        
        print(f'Random Forest - Validation Accuracy: {rf_val_acc:.4f}, Test Accuracy: {rf_test_acc:.4f}')
    
    case _:
        raise ValueError(f"Invalid model type: {model_type}")
