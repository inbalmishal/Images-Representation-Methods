import numpy as np
import tensorflow as tf

from keras import layers, losses
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Flatten


class Autoencoder_28(Model):
    """
    used for mnist and mnist fashion
    """
    def __init__(self, latent_dim):
        super(Autoencoder_28, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            Flatten(),
            Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            Dense(28 ** 2, activation='sigmoid'),
            Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_32(Model):
    """
    used for cifar10
    """
    def __init__(self, latent_dim):
        super(Autoencoder_32, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Flatten(),
            Dense(self.latent_dim, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            Dense(4*4*self.latent_dim, activation='relu'),
            Reshape((4, 4, self.latent_dim)),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

