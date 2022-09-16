import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import tensor
from tqdm import tqdm
import classifier
import fasttext

model_FastTest = fasttext.load_model('../model/cc.en.300.bin')
model_net = classifier.IdeenAutoEncoder(input=300, h1=2)
model_net.train_itself(model_FastTest=model_FastTest, epochs=250, batch_size=512, lr=0.01, delete_casual_words=True)

embeddings_from_autoencoder = []
labels_from_autoencoder = []

embeddings_from_autoencoder.extend(model_net.all_casual_embeddings.detach().numpy())
embeddings_from_autoencoder.extend(model_net.all_thematic_embeddings.detach().numpy())
labels_from_autoencoder.extend(["b"] * len(model_net.all_casual_embeddings.detach().numpy()))
labels_from_autoencoder.extend(["r"] * len(model_net.all_thematic_embeddings.detach().numpy()))

embeddings_from_autoencoder = np.asarray(embeddings_from_autoencoder)
labels_from_autoencoder = np.asarray(labels_from_autoencoder)

fig, axis = plt.subplots(figsize=(10, 10))
plt.scatter(embeddings_from_autoencoder[:, 0], embeddings_from_autoencoder[:, 1],
                       color=labels_from_autoencoder,
                       marker='^')
plt.savefig("data/" + "compressed_in_autoencoder_embeddings/2D embeddings from Autoencoder 2.0 V Test8.png", dpi=300)

plt.show()
