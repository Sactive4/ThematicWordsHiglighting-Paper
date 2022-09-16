import shutil
from sklearn.manifold import TSNE
import random
import work_with_files
import fasttext
import time
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import engine
from torch.utils.data import DataLoader
import fasttext.util
import os
from tqdm import tqdm
from torch import tensor
import classifier
from sklearn.decomposition import KernelPCA


def reduce_dimensionality(working_dir=None, version="RD5-2.1.2", n_components=5, delete_casual_words=False, load_all_words=False):
    """
    This function reduces dimension of data to *n_components* embeddings and draw their graph
    Then save them to "working_dir" + "-" + "version"!
    """
    start = time.perf_counter()

    if working_dir is None:
        working_dir = "compressed_embeddings-" + version
    else:
        working_dir = working_dir + "-" + version
    working_dir = "data/" + working_dir

    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir)

    print("Working_with_model was started!")

    # model_net = torch.load("../model/IdeenPerceptron-V3-7-with-0.901-F1-and.pth")
    model_FastTest = fasttext.load_model('../model/cc.en.300.bin')

    print("Models were imported!")

    thematic_words, casual_words = work_with_files.import_words(load_all_words=load_all_words)

    # getting compressed embeddings by tSNE model

    data = []
    labels = []
    data_words = []

    data.extend(list(engine.get_words_embeddings(thematic_words, model=model_FastTest)))
    labels.extend(["r"] * len(thematic_words))
    data_words.extend(thematic_words)

    if not delete_casual_words:
        data.extend(list(engine.get_words_embeddings(casual_words, model=model_FastTest)))
        labels.extend(["b"] * len(casual_words))
        data_words.extend(casual_words)

    triplets = list(zip(data, labels, data_words))
    # random.shuffle(triplets)
    data, labels, data_words = zip(*triplets)
    data = np.asarray(data)
    labels = np.asarray(labels)
    data_words = np.asarray(data_words)
    print("data.shape = " + str(data.shape))
    print("Import FINISHED in {} seconds time".format(time.perf_counter() - start))

    # perplexity_array = [400, 500, 750, 1000]
    perplexity_array = [10, 25, 50, 100]    # , 250, 300]

    tSNE_method = 'barnes_hut'
    if n_components > 3:
        tSNE_method = 'exact'

    X_embedded_1 = TSNE(n_components=n_components, perplexity=perplexity_array[0], learning_rate='auto', init='pca',
                        n_iter=5000, method=tSNE_method, n_jobs=-1).fit_transform(
        data)
    print("X_embedded_" + str(perplexity_array[0]) + " was computed in {} seconds time".format(
        time.perf_counter() - start))
    X_embedded_2 = TSNE(n_components=n_components, perplexity=perplexity_array[1], learning_rate='auto', init='pca',
                        n_iter=5000, method=tSNE_method, n_jobs=-1).fit_transform(data)
    print("X_embedded_" + str(perplexity_array[1]) + " was computed in {} seconds time".format(
        time.perf_counter() - start))
    X_embedded_3 = TSNE(n_components=n_components, perplexity=perplexity_array[2], learning_rate='auto', init='pca',
                        n_iter=5000, method=tSNE_method, n_jobs=-1).fit_transform(data)
    print("X_embedded_" + str(perplexity_array[2]) + " was computed in {} seconds time".format(
        time.perf_counter() - start))
    X_embedded_4 = TSNE(n_components=n_components, perplexity=perplexity_array[3], learning_rate='auto', init='pca',
                        n_iter=5000, method=tSNE_method, n_jobs=-1).fit_transform(data)
    print("X_embedded_" + str(perplexity_array[3]) + " was computed in {} seconds time".format(
        time.perf_counter() - start))
    print("X_embedded.shape = " + str(X_embedded_2.shape))

    # getting compressed embeddings by Autoencoder model

    model_net = classifier.IdeenAutoEncoder(input=300, h1=n_components)
    model_net.train_itself(model_FastTest=model_FastTest, epochs=200, batch_size=256, lr=0.01,
                           delete_casual_words=delete_casual_words, load_all_words=load_all_words)

    embeddings_from_autoencoder = []
    labels_from_autoencoder = []

    embeddings_from_autoencoder.extend(model_net.all_thematic_embeddings.detach().numpy())
    labels_from_autoencoder.extend(["r"] * len(model_net.all_thematic_embeddings.detach().numpy()))

    if not delete_casual_words:
        embeddings_from_autoencoder.extend(model_net.all_casual_embeddings.detach().numpy())
        labels_from_autoencoder.extend(["b"] * len(model_net.all_casual_embeddings.detach().numpy()))

    embeddings_from_autoencoder = np.asarray(embeddings_from_autoencoder)
    labels_from_autoencoder = np.asarray(labels_from_autoencoder)

    print("embeddings_from_autoencoder.shape = " + str(embeddings_from_autoencoder.shape))

    # getting compressed embeddings by PCA model

    transformer = KernelPCA(n_components=n_components, kernel='linear')
    X_embedded_by_linear_PCA = transformer.fit_transform(data)
    print("X_embedded_by_linear_PCA was computed")
    transformer = KernelPCA(n_components=n_components, kernel='poly')
    X_embedded_by_poly_PCA = transformer.fit_transform(data)
    print("X_embedded_by_poly_PCA was computed")
    transformer = KernelPCA(n_components=n_components, kernel='sigmoid')
    X_embedded_by_sigmoid_PCA = transformer.fit_transform(data)
    print("X_embedded_by_sigmoid_PCA was computed")
    transformer = KernelPCA(n_components=n_components, kernel='cosine')
    X_embedded_by_cosine_PCA = transformer.fit_transform(data)
    print("X_embedded_by_cosine_PCA was computed")
    transformer = KernelPCA(n_components=n_components, kernel='rbf')
    X_embedded_by_rbf_PCA = transformer.fit_transform(data)
    print("X_embedded_by_rbf_PCA was computed")

    if n_components == 2:
        fig, axis = plt.subplots(5, 2, figsize=(25, 40))  #
        axis[0, 0].scatter(embeddings_from_autoencoder[:, 0], embeddings_from_autoencoder[:, 1],
                           color=labels_from_autoencoder,
                           marker='^')
        axis[0, 0].set_title("embeddings from Autoencoder")

        axis[1, 0].scatter(X_embedded_1[:, 0], X_embedded_1[:, 1], color=labels, marker='^')
        axis[1, 0].set_title("embeddings from t-SNE, perplexity=" + str(perplexity_array[0]))

        axis[2, 0].scatter(X_embedded_2[:, 0], X_embedded_2[:, 1], color=labels, marker='^')
        axis[2, 0].set_title("embeddings from t-SNE, perplexity=" + str(perplexity_array[1]))

        axis[3, 0].scatter(X_embedded_3[:, 0], X_embedded_3[:, 1], color=labels, marker='^')
        axis[3, 0].set_title("embeddings from t-SNE, perplexity=" + str(perplexity_array[2]))

        axis[4, 0].scatter(X_embedded_4[:, 0], X_embedded_4[:, 1], color=labels, marker='^')
        axis[4, 0].set_title("embeddings from t-SNE, perplexity=" + str(perplexity_array[3]))

        axis[0, 1].scatter(X_embedded_by_linear_PCA[:, 0], X_embedded_by_linear_PCA[:, 1], color=labels,
                           marker='^')
        axis[0, 1].set_title("embeddings from linear-PCA")

        axis[1, 1].scatter(X_embedded_by_poly_PCA[:, 0], X_embedded_by_poly_PCA[:, 1], color=labels,
                           marker='^')
        axis[1, 1].set_title("embeddings from poly-PCA")

        axis[2, 1].scatter(X_embedded_by_sigmoid_PCA[:, 0], X_embedded_by_sigmoid_PCA[:, 1], color=labels,
                           marker='^')
        axis[2, 1].set_title("embeddings from sigmoid-PCA")

        axis[3, 1].scatter(X_embedded_by_cosine_PCA[:, 0], X_embedded_by_cosine_PCA[:, 1], color=labels,
                           marker='^')
        axis[3, 1].set_title("embeddings from cosine-PCA")

        axis[4, 1].scatter(X_embedded_by_rbf_PCA[:, 0], X_embedded_by_rbf_PCA[:, 1], color=labels,
                           marker='^')
        axis[4, 1].set_title("embeddings from rbf-PCA")

        plt.savefig(working_dir + "/2D embeddings visualization V2-1.1.pdf", dpi=300)
        plt.show()

    else:
        embeddings_array = [X_embedded_1, X_embedded_2, X_embedded_3, X_embedded_4, embeddings_from_autoencoder,
                            X_embedded_by_poly_PCA, X_embedded_by_rbf_PCA, X_embedded_by_cosine_PCA,
                            X_embedded_by_linear_PCA, X_embedded_by_sigmoid_PCA]
        embeddings_names_array = ["embeddings from t-SNE, perplexity=" + str(perplexity_array[0]),
                                  "embeddings from t-SNE, perplexity=" + str(perplexity_array[1]),
                                  "embeddings from t-SNE, perplexity=" + str(perplexity_array[2]),
                                  "embeddings from t-SNE, perplexity=" + str(perplexity_array[3]),
                                  "embeddings from Autoencoder",
                                  "embeddings from poly-PCA", "embeddings from rbf-PCA", "embeddings from cosine-PCA",
                                  "embeddings from linear-PCA", "embeddings from sigmoid-PCA"]

        thematic_size = 20
        casual_size = 10
        if delete_casual_words:
            casual_size = 0
            thematic_size = 30

        words_for_plot = np.append(data_words[:thematic_size], data_words[len(data_words) - casual_size:], axis=0)

        for embeddings_index in range(len(embeddings_array)):
            embeddings = embeddings_array[embeddings_index]
            embeddings_name = embeddings_names_array[embeddings_index]
            embeddings_for_plot = np.append(embeddings[:thematic_size], embeddings[len(embeddings) - casual_size:], axis=0)

            fig, ax = plt.subplots(figsize=(n_components+5, thematic_size+casual_size))
            ax.imshow(embeddings_for_plot)
            plt.yticks(list(range(len(words_for_plot))), words_for_plot)
            plt.savefig(working_dir + "/" + embeddings_name + ".jpg", dpi=300)


if __name__ == '__main__':
    reduce_dimensionality(delete_casual_words=False, load_all_words=False)
