
import re
from urllib import response
import tensorflow.python as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

BATCH_SIZE = 10
EPOCHS = 2
VERTEX_NULL_SPACE = -1.0
INDEX_NULL_SPACE = -1.0
ERROR = 0.5

FACE_COUNT = 100
DIMENSIONS = int( (.60 * FACE_COUNT) ** 0.33 )

reponse = input(f'Dimensions are {DIMENSIONS}. Would you like to change it? (yes or no)\n')
if response.lower() == 'yes':
    DIMENSIONS = int(input('Enter in your number for the dimensions\n'))

INDEX_COUNT = DIMENSIONS ** 3
PADDING = 1
DIMENSIONS += PADDING

class GAN(keras.Model):
    def __int__(self, vertex_latent_dims: int, index_latent_dims: int):

        self.vertex_generator = keras.Sequential([
            keras.layers.Input(shape=(vertex_latent_dims,)),
            layers.Dense(1 * 1 * 128),
            layers.Reshape((1, 1, 1, 128)),
            layers.Conv3DTranspose(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv3DTranspose(256, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv3DTranspose(512, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv3D(1, kernel_size=4, padding='same', activation='sigmoid')
        ])
        self.index_generator = keras.Sequential([
            keras.layers.Input(shape=(8, 8, 8, 3)),
            layers.Dense(3 * 3 * 128),
            layers.Reshape((3, 3, 128)),
            layers.Conv3D(64, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv3D(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv3D(256, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(50, activation='tanh'),
            layers.Dense(512, activation='sigmoid')
        ])

        self.vertex_discriminator = keras.Sequential([
            keras.layers.Input(shape=(8, 8, 8, 3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])
        self.index_discriminator = keras.Sequential([
            keras.layers.Input(shape=(8, 8, 8, 1)),
            layers.Conv3D(64, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])
    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass