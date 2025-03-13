""" 
author:-aam35
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

tf.executing_eagerly()
# Define paramaters for the model
learning_rate = 0.001
batch_size = 1000
n_epochs = 1000
n_train = 100
n_test = 100

# Step 1: Read in data
fmnist_folder = './data/fashion'
#Create dataset load function [Refer fashion mnist github page for util function]
#Create train,validation,test split
#train, val, test = utils.read_fmnist(fmnist_folder, flatten=True)

# https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_images, train_class = load_mnist(fmnist_folder)
train_data = tf.data.Dataset.from_tensor_slices((train_images, train_class)).batch(batch_size)

# create testing Dataset and batch it
test_images, test_class = load_mnist(fmnist_folder, kind="t10k")
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_class)).batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
img_shape = 28*28
num_classes = 10
w = tf.Variable(tf.random.normal([img_shape, num_classes], stddev=0.01))
b = tf.Variable(tf.zeros([num_classes]))

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
def model(x):
    return tf.matmul(x, w) + b

logits = model(img)

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#Step 8: train the model for n_epochs times
for i in range(n_epochs):
    total_loss = 0
    n_batches = 0
    #Optimize the loss function
    print("Train and Validation accuracy")
    for batch, (batch_images, batch_labels) in enumerate(train_data):
        with tf.GradientTape() as tape:
            batch_logits = model(batch_images)
            batch_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(batch_labels, num_classes), logits=batch_logits))
        grads = tape.gradient(batch_loss, [w, b])
        optimizer.apply_gradients(zip(grads, [w, b]))
        total_loss += batch_loss.numpy()
        n_batches += 1
    
    print(f"Epoch {i+1}, Loss: {total_loss / n_batches}")
	
#Step 9: Get the Final test accuracy
total_correct_preds = 0
total_samples = 0

for batch_images, batch_labels in test_data:
    batch_logits = model(batch_images)
    batch_preds = tf.nn.softmax(batch_logits)
    batch_correct_preds = tf.equal(tf.argmax(batch_preds, 1), tf.argmax(tf.one_hot(batch_labels, num_classes), 1))
    total_correct_preds += tf.reduce_sum(tf.cast(batch_correct_preds, tf.float32)).numpy()
    total_samples += batch_labels.shape[0]

test_accuracy = total_correct_preds / total_samples
print(f"Final Test Accuracy: {test_accuracy:.4f}")

#Step 10: Helper function to plot images in 3*3 grid
#You can change the function based on your input pipeline

def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Get image from test set 
images = test_data[0:9]

# Get the true classes for those images.
y = test_class[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y)


#Second plot weights 

def plot_weights(w=None):
    # Get the values for the weights from the TensorFlow variable.
    #TO DO ####
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = None
    #TO DO## obtains these value from W
    w_max = None

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

