
import tensorflow.python as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

BATCH_SIZE = 10
EPOCHS = 2
FACE_COUNT = 200
NULL_SPACE = -1.0
ERROR = 0.5


class GAN(keras.Model):
    def __int__(self, vertex_latent_dims: int, index_latent_dims: int):

        self.vertex_generator = keras.Sequential([
            keras.layers.Input(shape=(vertex_latent_dims,)),
            layers.Dense(3 * 3 * 128),
            layers.Reshape((3, 3, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=4, padding='same', activation='sigmoid')
        ])
        self.index_generator = keras.Sequential([
            keras.layers.Input(shape=(index_latent_dims,)),
            layers.Dense(3 * 3 * 128),
            layers.Reshape((3, 3, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=4, padding='same', activation='sigmoid')
        ])
        self.vertex_discriminator = keras.Sequential([

        ])
        self.index_discriminator = keras.Sequential([

        ])
    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass