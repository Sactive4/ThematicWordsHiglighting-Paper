import numpy as np

import classifier
import fasttext.util
import matplotlib.pyplot as plt


model_FastTest = fasttext.load_model('../model/cc.en.300.bin')

model_net = classifier.IdeenAutoEncoder(input=300, h1=2)
model_net.train_itself(model_FastTest=model_FastTest, epochs=100, batch_size=32, lr=0.01)

# data.extend(list(engine.get_words_embeddings(thematic_words, model=model_FastTest)))
# data.extend(list(engine.get_words_embeddings(casual_words, model=model_FastTest)))
embeddings_from_autoencoder = []
labels_from_autoencoder = []
embeddings_from_autoencoder.extend(model_net.all_casual_embeddings.detach().numpy())
embeddings_from_autoencoder.extend(model_net.all_thematic_embeddings.detach().numpy())
labels_from_autoencoder.extend(["b"] * len(model_net.all_casual_embeddings.detach().numpy()))
labels_from_autoencoder.extend(["r"] * len(model_net.all_thematic_embeddings.detach().numpy()))
embeddings_from_autoencoder = np.asarray(embeddings_from_autoencoder)
labels_from_autoencoder = np.asarray(labels_from_autoencoder)
print("embeddings_from_autoencoder.shape = " + str(embeddings_from_autoencoder.shape))

fig, ax = plt.subplots()
plt.scatter(embeddings_from_autoencoder[:, 0], embeddings_from_autoencoder[:, 1], color=labels_from_autoencoder, marker='^')
plt.show()