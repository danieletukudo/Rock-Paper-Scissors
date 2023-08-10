import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Building the AI model
model = tf.keras.models.Sequential([
    # First convolutional layer with 16 filters, a 3x3 kernel, and a ReLU activation function
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    # Max pooling layer with a 2x2 pool size
    tf.keras.layers.MaxPooling2D(2, 2),

    # Second convolutional layer with 32 filters, a 3x3 kernel, and a ReLU activation function
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # Max pooling layer with a 2x2 pool size
    tf.keras.layers.MaxPooling2D(2, 2),

    # Third convolutional layer with 64 filters, a 3x3 kernel, and a ReLU activation function
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Max pooling layer with a 2x2 pool size
    tf.keras.layers.MaxPooling2D(2, 2),

    # Fourth convolutional layer with 64 filters, a 3x3 kernel, and a ReLU activation function
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Max pooling layer with a 2x2 pool size
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the output from the convolutional layers into a 1D vector
    tf.keras.layers.Flatten(),
    # Dropout layer with a rate of 0.5 to prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # Dense layer with 512 units and a ReLU activation function
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer with 3 units and a softmax activation function (for multi-class classification)
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model with categorical cross-entropy loss, RMSprop optimizer, and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Define the data generators for training and validation
train_datagen = ImageDataGenerator(
    # Apply image augmentation techniques such as rescaling, rotation, shifting, shearing, and flipping
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_dir = 'train'
val_dir = 'validation'
val_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=126,
    # Since we use categorical cross-entropy loss, we need categorical labels
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,  # This is the source directory for validation images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=31,
    # Since we use categorical cross-entropy loss, we need categorical labels
    class_mode='categorical'
)

# Train the model on the training data for 40 epochs with validation data
history = model.fit(
    train_generator,
    steps_per_epoch=20,
    epochs=40,
    verbose=1,
    validation_data=val_generator,
    validation_steps=12
)

# Save the trained model to a file
model.save('model.h5')

#
