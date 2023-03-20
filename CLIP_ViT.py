########################################################################################################################
########################################################################################################################
########################################## You can find the original code here: ########################################
######## https://github.com/pinecone-io/examples/blob/master/learn/image-retrieval/vision-transformers/vit.ipynb #######
########################################################################################################################
########################################################################################################################


# pip install datasets transformers torch

import logging
import torch
from torchvision import datasets
data_path = '/'
from transformers import ViTFeatureExtractor
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm.auto import tqdm
import numpy as np
import os

def notebook_code():
    train_set = datasets.CIFAR10(data_path, train=True, download=True)
    test_set = datasets.CIFAR10(data_path, train=False, download=True)

    # if you have CUDA or MPS, set it to the active device like this
    device = "cuda" if torch.cuda.is_available() else \
        ("mps" if torch.backends.mps.is_available() else "cpu")
    model_id = "openai/clip-vit-base-patch16"

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    train_set_images = [im for (im, label) in train_set]
    test_set_images = [im for (im, label) in test_set]

    batch_size = 16
    train_arr = None
    test_arr = None

    # train set
    for i in tqdm(range(0, len(train_set_images), batch_size)):
        # select batch of images
        batch = train_set_images[i: i + batch_size]

        # process and resize
        batch = processor(
            text=None,
            images=batch,
            return_tensors='pt',
            padding=True
        )['pixel_values'].to(device)

        # get image embeddings
        batch_emb = model.get_image_features(pixel_values=batch)

        # convert to numpy array
        batch_emb = batch_emb.squeeze(0)
        batch_emb = batch_emb.cpu().detach().numpy()

        # add to larger array of all image embeddings
        if train_arr is None:
            train_arr = batch_emb
        else:
            train_arr = np.concatenate((train_arr, batch_emb), axis=0)

    # test set
    for i in tqdm(range(0, len(test_set_images), batch_size)):
        # select batch of images
        batch = test_set_images[i: i + batch_size]

        # process and resize
        batch = processor(
            text=None,
            images=batch,
            return_tensors='pt',
            padding=True
        )['pixel_values'].to(device)

        # get image embeddings
        batch_emb = model.get_image_features(pixel_values=batch)

        # convert to numpy array
        batch_emb = batch_emb.squeeze(0)
        batch_emb = batch_emb.cpu().detach().numpy()

        # add to larger array of all image embeddings
        if test_arr is None:
            test_arr = batch_emb
        else:
            test_arr = np.concatenate((test_arr, batch_emb), axis=0)

    # normalize
    train_arr = train_arr.T / np.linalg.norm(train_arr, axis=1)
    test_arr = train_arr.T / np.linalg.norm(test_arr, axis=1)

    # save
    path_train = 'representations_results/ViT/cifar10/train1.npy'
    path_test = 'representations_results/ViT/cifar10/test1.npy'

    if os.path.exists(path_train) and os.path.exists(path_test):
        print("files already exists")
        return

    if not os.path.isdir("representations_results/results"):
        os.makedirs("representations_results/results")
    if not os.path.isdir("representations_results/ViT"):
        os.makedirs("representations_results/ViT")
    if not os.path.isdir(f'representations_results/ViT/cifar10'):
        os.makedirs(f'representations_results/ViT/cifar10')

    # save the result
    if path_train:
        np.save(path_train, train_arr)

    if path_test:
        np.save(path_test, test_arr)