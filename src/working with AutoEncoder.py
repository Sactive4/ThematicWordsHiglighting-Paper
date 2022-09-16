import classifier
import work_with_files
import fasttext
import time
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from classifier import IdeenPerceptron
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

if __name__ == '__main__':
    start = time.perf_counter()

    model_FastTest = fasttext.load_model('../model/cc.en.300.bin')

    h1 = 3
    model_net = classifier.IdeenAutoEncoder(input=300, h1=h1)
    epochs = 100
    batch_size = 32
    device_name = "cpu"
    # device_name = "cuda:0"
    device = torch.device(device_name)
    model_net.to(device)
    lr = 0.01

    thematic_words, casual_words = work_with_files.import_words(load_all_words=True)
    print("Number of thematic words =", len(thematic_words), "Number of casual words =", len(casual_words))
    train_dataLoader, test_dataLoader, train_data_size, test_data_size = engine.generate_sets_from_words_for_autoencoder(
        thematic_words_list=thematic_words, casual_words_list=casual_words, batch_size=batch_size, model=model_FastTest, delete_casual_words=True)

    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model_net.parameters(), lr=lr)
    transform = transforms.ToTensor()

    start_learning = time.perf_counter()
    print("Learning was started!")

    accuracy_array = []

    for epoch in range(epochs):
        print("epoch № " + str(epoch) + "\n" + str("_" * 100))

        train_loss = 0

        for embedding, labels in tqdm(train_dataLoader):
            optimizer.zero_grad()

            embedding = embedding.to(device)
            labels = labels.to(device)
            output = model_net(embedding)
            # print(output, labels)
            # print(loss_function(output, labels))

            loss = loss_function(output, labels)
            # loss = nn.L1Loss(output, labels)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

        test_loss = 0
        correct = 0
        # true_positives = 0
        # false_and_true_positives = 0
        # relevant_items = 0

        for embedding, labels in tqdm(test_dataLoader):
            # print(embedding, labels)
            with torch.no_grad():
                embedding = embedding.to(device)
                labels = labels.to(device)
                output = model_net(embedding)
                if device_name == "cuda:0":
                    predictions = tensor(output.cpu().numpy().round(2)).to(device)
                else:
                    predictions = tensor(output.numpy().round(2))
                # print(output, labels)
                loss = loss_function(output, labels)
                # true_positives += (predictions[predictions == labels] == 1).sum()
                # false_and_true_positives += predictions.sum()
                # relevant_items += labels.sum()
                test_loss += loss.item()
                if device_name == "cuda:0":
                    correct += (predictions == tensor(labels.cpu().numpy().round(2)).to(device)).sum()
                else:
                    correct += (predictions == tensor(labels.numpy().round(2))).sum()

        print(correct)
        print(f"Train loss * 1000: {1000 * train_loss / train_data_size:.3f}")
        print(f"Test loss * 1000: {1000 * test_loss / test_data_size:.3f}")
        print(f"Test accuracy: {correct / test_data_size / 300:.3f}")
        # print(f"Test Recall: {float(true_positives / relevant_items) :.3f}")
        # print(f"Test Precision: {float(true_positives / false_and_true_positives) :.3f}")
        # print(f"Test F1 SCORE: {np.sqrt(float(true_positives**2 / relevant_items / false_and_true_positives)) :.3f}")

        if epoch >= 3:
            accuracy_array.append(round(float(correct / test_data_size / 300), 4))
            if accuracy_array[-1] >= 0.999:
                epochs = epoch + 1
                break

    finish_learning = time.perf_counter()
    print("Learning was finished in {} seconds time".format(finish_learning - start_learning))

    model_net.draw_embeddings_graphs_and_save_embeddings(number_of_layers=0, model_FastTest=model_FastTest, device_name=device_name)

    fig, ax = plt.subplots()
    plt.plot(list(range(3, epochs)), accuracy_array)
    plt.savefig("data/" + "autoencoder_accuracy.jpg")

    print("MAX accuracy = " + str(max(accuracy_array)))

    # print(train_data_size)
    # print(test_data_size)

    device = torch.device("cpu")
    model_net.to(device)

    torch.save(model_net, "../model/IdeenAutoEncoder-V1-" + str(h1) + "-with-" + str(
        round(max(accuracy_array), 3)) + "-accuracy.pth")

    finish = time.perf_counter()
    print("Full process FINISHED in {} seconds time".format(finish - start))
