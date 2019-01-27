import tensorflow as tf
from tensorflow import keras

import math
import numpy as np
import matplotlib.pyplot as plt

PATH_TRAINED_WEIGHTS = "trained_1/model_weights.ckpt"
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# pre processing the labels
train_labels = keras.utils.to_categorical(y=train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(y=test_labels, num_classes=10)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('Size of : ')
print('Train Images : {}'.format(train_images.shape))
print('Train Labels : {}'.format(train_labels.shape))
print('Test Images : {}'.format(test_images.shape))
print('Test Labels : {}'.format(test_labels.shape))

# pre-process the images -> Normalization (Z - score)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

mean = np.mean(train_images, axis=(0, 1, 2, 3))
std = np.std(train_images, axis=(0, 1, 2, 3))
train_images = (train_images - mean) / (std + 1e-7)
test_images = (test_images - mean) / (std + 1e-7)

print('Pre-processing train and test images : ')
print('Mean : {}'.format(mean))
print('Standard deviation : {}'.format(std))


def plot_image(images, labels):
    grid_size = math.ceil(math.sqrt(images.shape[0]))
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size)
    for i, axes in enumerate(axes.flat):
        if i < images.shape[0]:
            axes.imshow(images[i].reshape(32, 32, 3))
            axes.set_xlabel("Ground truth : {}".format(classes[np.argmax(labels[i])]))
            axes.set_xticks([])
            axes.set_yticks([])
    plt.show()


plot_image(train_images[:16], train_labels[:16])

# create the model
weight_decay = 1e-4
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[32, 32, 3]),
    keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    ),
    keras.layers.Activation(activation=tf.nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    ),
    keras.layers.Activation(activation=tf.nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=[2, 2]),
    keras.layers.Dropout(rate=0.2),

    keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    ),
    keras.layers.Activation(activation=tf.nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    ),
    keras.layers.Activation(activation=tf.nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=[2, 2]),
    keras.layers.Dropout(rate=0.3),

    keras.layers.Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    ),
    keras.layers.Activation(activation=tf.nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    ),
    keras.layers.Activation(activation=tf.nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=[2, 2]),
    keras.layers.Dropout(rate=0.4),

    keras.layers.Flatten(),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# create a data generator to pre-process the images before forward propagation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(train_images)

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
    metrics=['accuracy']
)


def lr_schedule(epoch):
    """
    Function for LearningRateScheduler to take input as @epoch and
    return a new learning rate as per the epoch index value

    Decrease the learning rate as the number of epochs become
    large

    :param epoch: epoch index
    :return: new learning rate
    """
    learning_rate = 0.001
    if epoch > 75:
        learning_rate = 0.0005
    if epoch > 100:
        learning_rate = 0.0003
    if epoch > 125:
        learning_rate = 0.0001
    return learning_rate


learning_rate_scheduler = keras.callbacks.LearningRateScheduler(
    schedule=lr_schedule,
    verbose=1
)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    PATH_TRAINED_WEIGHTS,
    save_weights_only=True,
    verbose=1
)

# load previously trained weights
model.load_weights(PATH_TRAINED_WEIGHTS)

"""
batch_size = 64
model.fit_generator(
    datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=train_images.shape[0] / batch_size,
    epochs=150,
    initial_epoch=150,
    verbose=1,
    workers=6,
    validation_data=[test_images, test_labels],
    callbacks=[learning_rate_scheduler, model_checkpoint]
)
# """

# save to disk
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy : {}'.format(test_acc * 100))


def plot_cnn_weights(weights, input_channel=0):
    """
    Visualizes filters of the given weights
    :param weights: Weight of the convolutional layer, list of len = 4
    :param input_channel: Channel for which the filters are to be visualized
    """
    # Get lowest and highest weights
    weight_min = np.min(weights)
    weight_max = np.max(weights)
    num_filters = weights.shape[3]

    num_rows = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_rows)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            image = weights[:, :, input_channel, i]
            ax.imshow(
                image,
                vmin=weight_min,
                vmax=weight_max,
                interpolation='nearest',
                cmap='seismic',
            )
            ax.set_xlabel('Filter {}'.format(i))

        ax.set_xticks([])
        ax.set_yticks([])

    # fig.suptitle('Filters for Input Channel {}'.format(input_channel))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


model_weights = model.get_weights()
plot_cnn_weights(weights=model.get_weights()[0], input_channel=0)
plot_cnn_weights(weights=model.get_weights()[2], input_channel=0)
