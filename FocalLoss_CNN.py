#author Matthew Cattaneo
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from focal_loss import SparseCategoricalFocalLoss 

# Define the filtering function
def filter_classes(features):
    # Retrieve the labels
    label = features['labels']
    # Check if any of the selected classes is present in the labels
    return tf.reduce_any([tf.reduce_any(label == class_name) for class_name in selected_classes], axis=0)

# Method which preprocesses the images for training and testing.  First, we resize the image to 256x256
#next, we normalize all pixel values between 0 and 1.  Finally, we apply a gaussian filter to reduce noise
#afterwards, for each image, 
def preprocess(features):
    # Retrieve the image
    image = features['image']
    # Resize the image to a specific size
    image = tf.image.resize(image, [256, 256])
    # Normalize the pixel values to the range of [0, 1]
    image = image / 255.0
    # Apply Gaussian noise reduction to the image
    image = tfa.image.gaussian_filter2d(image)
    # Retrieve the labels
    label = features['labels']
    # Get the integer index for the class, are reducing the labels per image to the first one (max class) of the selected classes
    label = tf.argmax(tf.reduce_any([label == class_name for class_name in selected_classes], axis=0))
    return image, label

# Load the dataset information, to obtain the label names
# we need to get the integer values for the classes, so we can extact only the images for the classes we are viewing
ds_info = tfds.builder('voc/2012').info
label_names = ds_info.features['labels'].names

# Specify the names of classes we are going to be training and testing on, in this case: cat, dog, aeroplane, car bicycle
selected_class_names = ['cat', 'dog', 'aeroplane', 'car', 'bicycle']
selected_classes = [label_names.index(name) for name in selected_class_names]

# Load the dataset using tensorflow
(training_dataset, testing_dataset), ds_info = tfds.load('voc/2012', split=['train', 'validation'], shuffle_files=True, with_info=True, as_supervised=False)

# Filter the dataset based on the specified classes from above
training_dataset = training_dataset.filter(filter_classes)
testing_dataset = testing_dataset.filter(filter_classes)

training_dataset = training_dataset.map(preprocess)
testing_dataset = testing_dataset.map(preprocess)

#Define layers for the model, pool size, input shape, and activation function for each layer 
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(selected_classes), activation='softmax')
])

# Compile the model using Sparse Categorical Focal Loss as the loss function and Adam Optimization
model.compile(optimizer='adam', loss=SparseCategoricalFocalLoss(gamma=2), metrics=['accuracy'])
print("Training Model")
# Train the model on the preprocessed training dataset
model.fit(training_dataset.batch(32), epochs=10)
print("Testing Model")
# Evaluate the trained model on the preprocessed testing dataset
model.evaluate(testing_dataset.batch(32))
