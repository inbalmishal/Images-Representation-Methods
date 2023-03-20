import os

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator

import warnings

warnings.simplefilter('ignore', category=DeprecationWarning)


def choose_optimal_k(data, plot=True):
    """
    choose the best k for PCA.
    """
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)

    cum_exp_var = np.cumsum(pca.explained_variance_)

    kn = KneeLocator(range(1, data.shape[1] + 1), list(pca.explained_variance_), curve='convex', direction='decreasing')

    if plot:
        plt.title("PCA of simCLR cifar10_pca - choosing the optimal k")
        plt.bar(range(0, data.shape[1]), pca.explained_variance_, align='center')
        plt.step(range(0, data.shape[1]), pca.explained_variance_, where='mid', color='black')
        plt.ylabel('Explained variance')
        plt.xlabel('PCA feature')
        xticks = list(range(0, data.shape[1], 100))
        xticks.append(kn.elbow)
        plt.xticks(ticks=xticks)
        plt.grid()
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red', label=f"opt_k = {kn.elbow}")
        plt.show()

    return kn.elbow


def perform_PCA(data, k=0.8, path_to_save=None):
    pca = PCA(n_components=k)
    pca.fit(data)
    reduced = pca.transform(data)
    print(f"reduced_data.shape={reduced.shape}")
    print('======================')

    # save the result
    if path_to_save:
        np.save(path_to_save, reduced)

    return reduced, k


def optimal_PCA(data, path_to_save=None, scaling=False, plot=True):
    """
    performs PCA algorithm on the data using the optimal k

    @param data: the original dataset
    @return: the new dataset after performing PCA
    """
    print('======================')
    print("Start optimal PCA")
    print('======================')

    # scale between 0 and 1
    if scaling:
        scaler = MinMaxScaler()
        data_rescaled = scaler.fit_transform(data)
    else:
        data_rescaled = data

    # find the best k for PCA
    k_opt = choose_optimal_k(data_rescaled, plot=plot)
    print('the optimal k: ', k_opt)
    print('======================')

    # perform PCA algorithm with the optimal k
    return perform_PCA(data_rescaled, k_opt, path_to_save)


def pca(path_train, path_test, inf="unknown", plot=True):
    train = np.load(path_train)
    test = np.load(path_test)
    data = np.concatenate((train, test), axis=0)
    print(f"data loaded. data.shape={data.shape}")
    print('============================================================================================')

    _, k = optimal_PCA(data, plot=plot)
    path = f'results/PCA/{inf}/'
    if os.path.exists(path + 'train1.npy') and os.path.exists(path + 'test1.npy'):
        print("files already exists")
        return

    if not os.path.isdir("representations_results/results"):
        os.makedirs("representations_results/results")
    if not os.path.isdir("representations_results/PCA"):
        os.makedirs("representations_results/PCA")
    if not os.path.isdir(f'results/PCA/{inf}'):
        os.makedirs(f'results/PCA/{inf}')


    perform_PCA(train, k, path + 'train1.npy')
    perform_PCA(test, k, path + 'test1.npy')

if __name__ == '__main__':
    path_train = 'representations_results/simCLR/cifar10/features_seed1.npy'
    path_test = 'representations_results/simCLR/cifar10/test_features_seed1.npy'

    pca(path_train, path_test, inf='simCLR_cifar10_scaled_k42', plot=True)


