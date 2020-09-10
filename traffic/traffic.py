import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # get the gtsrb folder path
    gtsrb_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
    # create lists for return
    folders_list = []
    images_list = []
    # with the name of each subfolder of gtsrb, iterate through them and get into the folder
    gtsrb = os.listdir(gtsrb_folder)
    for folder in gtsrb:
        gtsrb_sub = os.listdir(os.path.join(gtsrb_folder, folder))
        # iterate through each image of gtsrb subfolder and convert it to a numpy array, and finally append it to list
        # of images. Also append the name of each subfolder of gtsrb
        for image in gtsrb_sub:
            folders_list.append(int(folder))
            image_path = os.path.join(os.path.join(gtsrb_folder, folder), image)
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            dim = (IMG_WIDTH, IMG_HEIGHT)
            images_list.append(cv2.resize(img, dim, interpolation=cv2.INTER_AREA))

    return images_list, folders_list


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # create a convolutional neural network model
    # Worked the best so far with 10 and 130 filters.
    # A third layer of convolution doesnt perform quite well and may lead to overfitting

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            10, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            130, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    # model.summary helps to see how the image gets reduced in size along the network
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()
