import numpy as np

# from tensorflow import keras
# import tensorflow as tf
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.datasets import fashion_mnist, mnist, cifar10
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import os

from AE_model import Autoencoder_28, Autoencoder_32

transform = transforms.ToTensor()

# =====================================================================================================================
# ===================================================== datasets ======================================================
# =====================================================================================================================
def load_fashion_mnist():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    print(x_train.shape)
    print(x_test.shape)
    return x_train, x_test


def load_mnist():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    print(x_train.shape)
    print(x_test.shape)
    return x_train, x_test


def load_cifar10():
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape(len(x_train), x_train.shape[1], x_train.shape[2], 3)
    x_test = x_test.reshape(len(x_test), x_test.shape[1], x_test.shape[2], 3)
    print(x_train.shape)
    print(x_test.shape)
    return x_train, x_test
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================


def create_trained_model(x_train, x_test, path, SIZE, latent_dim):
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    if not os.path.exists(parent_path):
        if SIZE == 28:
            autoencoder = Autoencoder_28(latent_dim=64)
            autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
            autoencoder.summary()
            autoencoder.fit(x_train, x_train,
                            epochs=10,
                            shuffle=True,
                            validation_data=(x_test, x_test))
        elif SIZE == 32:
            autoencoder = Autoencoder_32(latent_dim=latent_dim)
            autoencoder.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
            autoencoder.build((None, SIZE, SIZE, 3))
            autoencoder.summary()
            autoencoder.fit(x_train, x_train,
                            epochs=30,
                            batch_size=256,
                            shuffle=True,
                            validation_data=(x_test, x_test))
        else:
            raise ("ENTER SIZE!")

        autoencoder.save_weights(path)

    elif SIZE == 28:
        autoencoder = Autoencoder_28(latent_dim=64)
        autoencoder.load_weights(path)
    elif SIZE == 32:
        autoencoder = Autoencoder_32(latent_dim=latent_dim)
        autoencoder.load_weights(path)

    return autoencoder


def plot_results(autoencoder, x_test, SIZE, latent_dim):
    print(f"latent_dim={latent_dim}")
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        if SIZE == 28:
            plt.imshow(x_test[i], cmap='gray')
        else:
            plt.imshow(x_test[i])
        plt.title("original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        if SIZE == 28:
            plt.imshow(decoded_imgs[i], cmap='gray')
        else:
            plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def save_codes(autoencoder, x, path, SIZE):
    if SIZE == 28:
        encoded_imgs = autoencoder.encoder(x).numpy()
        parent_path = os.path.abspath(os.path.join(path, os.pardir))
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        np.save(path, encoded_imgs)
    elif SIZE == 32:
        encoded_imgs = autoencoder.encoder(x).numpy()
        parent_path = os.path.abspath(os.path.join(path, os.pardir))
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        np.save(path, encoded_imgs)


def main(db_name):
    """
    example for creating AE representation
    """
    if db_name == "fashion_mnist":
        SIZE = 28
        x_train, x_test = load_fashion_mnist()
    elif db_name == "mnist":
        SIZE = 28
        x_train, x_test = load_mnist()
    elif db_name == "cifar10":
        SIZE = 32
        x_train, x_test = load_cifar10()
    else:
        raise "dataset not exists"

    latent_dim = int(SIZE**2/4)
    model_path = f"./models/model_{db_name}_lat{latent_dim}/my_checkpoint"
    autoencoder = create_trained_model(x_train, x_test, model_path, SIZE, latent_dim=latent_dim)
    plot_results(autoencoder, x_test, SIZE, latent_dim)
    embedded_data_path_train = f"./results/embedded_{db_name}_lat{latent_dim}_train1.npy"
    embedded_data_path_test = f"./results/embedded_{db_name}_lat{latent_dim}_test1.npy"
    save_codes(autoencoder, x_train, embedded_data_path_train, SIZE)
    save_codes(autoencoder, x_test, embedded_data_path_test, SIZE)


if __name__ == '__main__':
    main("cifar10")
