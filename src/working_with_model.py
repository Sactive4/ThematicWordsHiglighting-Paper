import random
import sys

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
import classifier


def classify_words_from_fasttext(model_name, model_FastText=None, classify_all_words_and_save_them=True,
                                 number_of_classified_words=-1, data_dir=None, version="V-1.1.1", load_all_words=False):
    start = time.perf_counter()
    print("Working_with_model was started!")

    if data_dir is None:
        data_dir = ""
    else:
        data_dir += "/"

    model_net = torch.load(data_dir + model_name)

    if model_FastText is None:
        model_FastText = fasttext.load_model('../model/cc.en.300.bin')

    print("Models were imported!")

    sys.stdout = open(data_dir + "output_working_with_model.txt", 'w')

    thematic_words, casual_words = work_with_files.import_words(load_all_words=load_all_words)

    print("_" * 100 + "\nWrong classified thematic words")
    thematic_mistakes_counter = 0

    for word in thematic_words:
        if model_net(tensor(engine.get_words_embeddings([word], model_FastText))).round() != 1:
            print(word)
            thematic_mistakes_counter += 1

    print("Recall = " + str(1 - thematic_mistakes_counter / len(thematic_words)) + "\n" + str(
        "_" * 100) + "\nWrong classified casual words")

    casual_mistakes_counter = 0
    for word in casual_words:
        if model_net(tensor(engine.get_words_embeddings([word], model_FastText))).round() != 0:
            print(word)
            casual_mistakes_counter += 1

    print("Precision = " + str((len(thematic_words) - thematic_mistakes_counter) / (
            len(thematic_words) - thematic_mistakes_counter + casual_mistakes_counter)) + "\n" + str("_" * 100))

    finish = time.perf_counter()
    print("Computing F1 finished in {} seconds time".format(finish - start))
    print("_" * 100)

    if not classify_all_words_and_save_them:
        return "Done"

    words = model_FastText.words
    clear_words = []
    for word in tqdm(words):
        if "." not in word and "," not in word and "1" not in word and "2" not in word and "3" not in word and "4" not in word and "5" not in word and "6" not in word and "7" not in word and "8" not in word and "9" not in word and "0" not in word:
            clear_words.append(word)

    words = clear_words
    random.shuffle(words)
    clear_words = None

    all_thematic_words = []
    all_casual_words = []

    start_predicting = time.perf_counter()

    if number_of_classified_words == -1:
        number_of_classified_words = len(words)

    predictions = model_net(tensor(engine.get_words_embeddings(words[:number_of_classified_words], model_FastText)))

    finish_predicting = time.perf_counter()
    print("\nPredicting was FINISHED in {} seconds time".format(finish_predicting - start_predicting))

    for word_index in tqdm(range(number_of_classified_words)):
        word = words[word_index]
        prediction = predictions[word_index]
        if round_tensor(tenor=prediction, round_threshold=0.3) == 1:
            all_thematic_words.append(word)
        else:
            all_casual_words.append(word)

    print("Thematic array size = " + str(len(all_thematic_words)))
    print("Casual array size = " + str(len(all_casual_words)))

    file = open(data_dir + str(number_of_classified_words) + " thematic words-" + version + "-.txt", "w",
                encoding="utf-8")
    file.write("\n".join(all_thematic_words))
    file.close()

    file = open(data_dir + str(number_of_classified_words) + " casual words-" + version + "-.txt", "w",
                encoding="utf-8")
    file.write("\n".join(all_casual_words))
    file.close()

    finish = time.perf_counter()
    print("Full process FINISHED in {} seconds time".format(finish - start))


def draw_embeddings_graphs():
    start = time.perf_counter()
    print("Working_with_model was started!")

    model_net = torch.load("../model/IdeenPerceptron-V3-7-with-0.901-F1-and.pth")
    model_FastTest = fasttext.load_model('../model/cc.en.300.bin')
    model_FastTest7 = fasttext.load_model('../model/cc.en.7.bin')

    print("Models were imported!")

    thematic_words, casual_words = work_with_files.import_words()
    random.shuffle(thematic_words)
    random.shuffle(casual_words)

    words_list = []
    words_list.extend(casual_words[:5])
    words_list.extend(thematic_words[:5])
    thematic_embeddings_size = 20

    parameters = []
    for param in model_net.parameters():
        parameters.append(param)

    net = classifier.IdeenPerceptronForGettingEmbedding(
        custom_weight=[tensor(parameters[0], dtype=torch.double), tensor(parameters[1], dtype=torch.double)])

    embeddings = engine.get_words_embeddings(words_list, model_FastTest)
    embeddings7 = engine.get_words_embeddings(words_list, model_FastTest7)
    compress_embeddings = net(tensor(embeddings, dtype=torch.double))
    thematic_embeddings = net(
        tensor(engine.get_words_embeddings(thematic_words[:thematic_embeddings_size], model_FastTest),
               dtype=torch.double))

    fig, ax = plt.subplots()
    ax.imshow(embeddings)
    plt.yticks(list(range(len(words_list))), words_list)
    plt.savefig("data/" + "embeddings.jpg")

    fig, ax = plt.subplots()
    ax.imshow(embeddings7)
    plt.yticks(list(range(len(words_list))), words_list)
    plt.savefig("data/" + "embeddings7.jpg")

    fig, ax = plt.subplots()
    ax.imshow(compress_embeddings.detach().numpy())
    plt.yticks(list(range(len(words_list))), words_list)
    plt.savefig("data/" + "compress_embeddings.jpg")

    fig, ax = plt.subplots()
    ax.imshow(thematic_embeddings.detach().numpy())
    plt.yticks(list(range(thematic_embeddings_size)), thematic_words[:thematic_embeddings_size])
    plt.savefig("data/" + "thematic_embeddings.jpg")

    finish = time.perf_counter()
    print("Full process FINISHED in {} seconds time".format(finish - start))


def round_tensor(tenor, round_threshold):
    auxiliary_array = []
    if len(tenor.size()) == 1:
        for element in tenor:
            if element >= round_threshold:
                auxiliary_array.append(1)
            else:
                auxiliary_array.append(0)
    elif len(tenor.size()) == 2:
        for line in tenor:
            auxiliary_array.append([])
            for element in line:
                if element >= round_threshold:
                    auxiliary_array[-1].append(1)
                else:
                    auxiliary_array[-1].append(0)

    return tensor(auxiliary_array, dtype=torch.float64)


def investigate_characteristics_of_the_model(model_net, model_fatText, test_dataLoader):
    device = torch.device("cpu")
    # device = torch.device("cuda:0")
    model_net.to(device)

    number_of_steps = 300

    precision = []
    recall = []
    accuracy = []
    F1_score = []
    threshold_array = list(range(number_of_steps + 1))

    for step in tqdm(threshold_array):  # Computing F1 for different threshold
        threshold = step / number_of_steps

        # test_loss = 0
        correct = 0
        true_positives = 0
        false_and_true_positives = 0
        relevant_items = 0

        for embedding, labels in test_dataLoader:
            with torch.no_grad():
                embedding = embedding.to(device)
                labels = labels.to(device)
                output = model_net(embedding)
                # predictions = output.round()
                predictions = round_tensor(output, threshold)
                predictions.to(device)
                # print(output,predictions, labels)
                true_positives += (predictions[predictions == labels] == 1).sum()
                false_and_true_positives += predictions.sum()
                relevant_items += labels.sum()
                # test_loss += loss.item()
                correct += (predictions == labels).sum()

        # print(f"Train loss: {train_loss / train_data_size:.3f}")
        # print(f"Test loss: {test_loss / test_data_size:.3f}")
        # print(f"Test accuracy: {correct / len(test_dataLoader):.3f}")
        accuracy.append(correct / len(test_dataLoader))
        # print(f"Test Recall: {float(true_positives / relevant_items) :.3f}")
        recall.append(float(true_positives / relevant_items))
        # print(f"Test Precision: {float(true_positives / false_and_true_positives) :.3f}")
        precision.append(float(true_positives / false_and_true_positives))
        # print(
        #     f"Test F1 SCORE: {np.sqrt(float(true_positives ** 2 / relevant_items / false_and_true_positives)) :.3f}"
        F1_score.append(np.sqrt(float(true_positives ** 2 / relevant_items / false_and_true_positives)))

    # Plotting Precision and Recall functions
    fig, axis = plt.subplots(2, figsize=(10, 10))

    axis[0].plot(threshold_array, precision, color='r', label='precision')
    axis[0].plot(threshold_array, recall, color='g', label='recall')
    axis[0].set_xlabel("threshold")
    axis[0].set_ylabel("Magnitude")
    axis[0].set_title("Precision and Recall functions")
    axis[0].legend()
    # plt.plot(threshold_array, precision, color='r', label='precision')
    # plt.plot(threshold_array, recall, color='g', label='recall')
    # plt.xlabel("threshold")
    # plt.ylabel("Magnitude")
    # plt.title("Precision and Recall functions")
    axis[1].plot(threshold_array, F1_score, color='b', label='F1 score')
    axis[1].set_xlabel("threshold")
    axis[1].set_ylabel("Magnitude")
    axis[1].set_title("F1 score function")
    axis[1].legend()

    plt.savefig("data/" + "F1_score_function.pdf", dpi=300)
    plt.show()
    # print("Max F1 score = " + str(max(F1_score).round(4)))


if __name__ == '__main__':
    # batch_size = 32
    # model_net = torch.load("WorkingWithModel-V-2.1.1/IdeenPerceptron-V-2.1.1-3-with-0.947-F1-and.pth")
    # model_FastText = fasttext.load_model('../model/cc.en.300.bin')

    # thematic_words, casual_words = work_with_files.import_words(load_all_words=True)
    # print("Number of thematic words =", len(thematic_words), "Number of casual words =", len(casual_words))
    # train_dataLoader, test_dataLoader, train_data_size, test_data_size = engine.generate_sets_from_words(
    #     thematic_words_list=thematic_words, casual_words_list=casual_words, batch_size=batch_size, model=model_FastTest,
    #     train_data_part=0.1)
    # investigate_characteristics_of_the_model(model_net=model_net, model_fatText=model_FastTest,
    #                                          test_dataLoader=test_dataLoader)
    classify_words_from_fasttext(model_name="IdeenPerceptron-Vh2-2.-1.3-50-with-0.836-F1-and.pth",
                                 # model_FastText=None,
                                 classify_all_words_and_save_them=True, number_of_classified_words=5000,
                                 data_dir="data/working_with_NN_classifier-Vh2-2.-1.3", version="V-1.1.1",
                                 load_all_words=True)
