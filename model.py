import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt




# Building the AI model
# Sequential means layers stacked one on another
model = tf.keras.models.Sequential([

    # First convolutional layer with 16 filters, a 3x3 kernel, and a ReLU activation function
# Convolution filters are filters (multi-dimensional data) used in Convolution layer which helps
#  in extracting specific features from input data.

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),

    # Max pooling layer with a 2x2 pool size
# Max pooling is a pooling operation that selects the maximum element from the region of the feature map covered by the filter.
    # Thus, the output after max-pooling layer would be a feature map containing the
    # most prominent features of the previous feature map.
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
    # Dropout layer with a rate of 0.2 to prevent overfitting

    # tf.keras.layers.Dropout(0.2),

    # Dense layer with 512 units and a ReLU activation function
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer with 3 units and a softmax activation function (for multi-class classification)

# Softmax | What is Softmax Activation Function | Introduction ...
# The softmax function is a mathematical function that converts a vector of real numbers into a probability distribution.

    tf.keras.layers.Dense(15, activation='softmax')
])

# Compile the model with categorical cross-entropy loss, RMSprop optimizer, and accuracy metric

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
# Define the data generators for training and validation
train_datagen = ImageDataGenerator(

    # augmentation  helps us to make the small dataset we have more effective
    # Apply image augmentation techniques such as rescaling, rotation, shifting, shearing, and flipping
   #  this is alays done to prevent overfiting imagine an image of a cat which is facing side and out model was not trained with that
   #  with augmentation we can rotate the image in a way which our model can use in training that and voiding over fitting
   # As rightly pointed out by you the rescale=1./255 will convert the pixels in range [0,255] to range [0,1].
    # This process is also called Normalizing the input. Scaling every dataset to the same range [0,1]
    # will make dataset contributes more evenly to the total loss.
    rescale=1/255,
    rotation_range=10,
    # shit the image around
    width_shift_range=0.1,
    height_shift_range=0.1,
    # kinda try to flip the image small
    shear_range=0.2,
    zoom_range=0.2,
    # flip the image horizontally
    horizontal_flip=True,
    # fill in the missing pixels with the nearest pixel
    fill_mode='nearest'
)

train_dir = 'image_data/train'
val_dir = 'image_data/val'
val_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(

    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=39,
    # Since we use categorical cross-entropy loss, we need categorical labels
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,  # This is the source directory for validation images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=18,
    # Since we use categorical cross-entropy loss, we need categorical labels
    class_mode='categorical'
)

# Train the model on the training data for 40 epochs with validation data
history = model.fit(
    train_generator,
    steps_per_epoch=21,
    epochs=40,

    # verbose parameter specifies how much to /
    # display while training is going on. With verbose set to 2, we'll get a little less animation hiding the epoch progress.

    verbose=1,
    validation_data=val_generator,
    validation_steps=8
)

# # Save the trained model to a file


model.save('model.h5')
# Check if the variable 'plot_history' is set to True
plot_history = True

# If 'plot_history' is False, do nothing (pass)
if plot_history == False:
    pass

# If 'plot_history' is True, execute the following code
else:
    # Extract the training accuracy history from the 'history' object
    acc = history.history['accuracy']

    # Extract the validation accuracy history from the 'history' object
    val_acc = history.history['val_accuracy']

    # Extract the training loss history from the 'history' object
    loss = history.history['loss']

    # Extract the validation loss history from the 'history' object
    val_loss = history.history['val_loss']

    # Create a range of values representing the number of training epochs
    epochs = range(len(acc))

    # Plot the training accuracy curve
    plt.plot(epochs, acc,label='Training Accuracy')

    # Plot the validation accuracy curve
    plt.plot(epochs, val_acc,label='Validation Accuracy')

    # Set the title of the accuracy plot
    plt.title('Training and Validation Accuracy')

    # Create a new figure for the next plot
    plt.figure()

    # Plot the training loss curve
    plt.plot(epochs, loss,label='Training Loss')

    # Plot the validation loss curve
    plt.plot(epochs, val_loss,label='Validation Loss')

    # Set the title of the loss plot
    plt.title('Training and Validation Loss')

    # Create a new figure for the next plot
    plt.figure()
