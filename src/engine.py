import fasttext
import time
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import work_with_files
from torch.utils.data import DataLoader
from torch import tensor


# model = fasttext.load_model('../model/cc.en.300.bin')


def get_words_embeddings(words, model):
    embeddings = []
    for words in words:
        embeddings.append(model.get_word_vector(words))
    return np.asarray(embeddings)


def round_nd_array(array, round_threshold):
    auxiliary_array = []
    if len(array.shape) == 1:
        for element in array:
            if element >= round_threshold:
                auxiliary_array.append(1)
            else:
                auxiliary_array.append(0)
    elif len(array.shape) == 2:
        for line in array:
            auxiliary_array.append([])
            for element in line:
                if element >= round_threshold:
                    auxiliary_array[-1].append(1)
                else:
                    auxiliary_array[-1].append(0)

    return np.asarray(auxiliary_array, dtype=np.float)


def intersect_with_model_words(words_array, model):
    model_words = np.asarray(model.words)
    return np.intersect1d(words_array, model_words)


def generate_sets_from_words(thematic_words_list, casual_words_list, batch_size, model=None, train_data_part=0.8,
                             repetition_of_thematic_selection=7, validation_data_part=0.0):
    start = time.perf_counter()
    print("Set generation was started")

    parts_for_train = [thematic_words_list[:int(len(thematic_words_list) * train_data_part)],
                       casual_words_list[:int(len(casual_words_list) * train_data_part)]]

    parts_for_test = [thematic_words_list[int(len(thematic_words_list) * train_data_part):int(
        len(thematic_words_list) * (1 - validation_data_part))],
                      casual_words_list[int(len(casual_words_list) * train_data_part):int(
                          len(casual_words_list) * (1 - validation_data_part))]]

    # if validation_data_part != 0.0:
    #     parts_for_validation = parts_for_test = [
    #         thematic_words_list[int(len(thematic_words_list) * (1 - validation_data_part)):],
    #         casual_words_list[int(len(thematic_words_list) * (1 - validation_data_part)):]]
    
    # Data was divided
    
    full_words_array_for_train = np.concatenate((parts_for_train[1], parts_for_train[0]))
    labels_for_train = np.concatenate((np.zeros(len(parts_for_train[1])), np.ones(len(parts_for_train[0]))))
    for i in range(repetition_of_thematic_selection):
        full_words_array_for_train = np.concatenate((full_words_array_for_train, parts_for_train[0]))
        labels_for_train = np.concatenate((labels_for_train, np.ones(len(parts_for_train[0]))))

    data_for_train = []
    for x, y in zip(full_words_array_for_train, labels_for_train):
        data_for_train.append([x, y])
    random.shuffle(data_for_train)

    X_for_train = []
    Y_for_train = []
    for pair in data_for_train:
        X_for_train.append(pair[0])
        Y_for_train.append([pair[1]])

    # Test list was generated

    full_words_array_for_test = np.concatenate((parts_for_test[1], parts_for_test[0]))
    labels_for_test = np.concatenate((np.zeros(len(parts_for_test[1])), np.ones(len(parts_for_test[0]))))

    data_for_test = []
    for x, y in zip(full_words_array_for_test, labels_for_test):
        data_for_test.append([x, y])
    random.shuffle(data_for_test)

    X_for_test = []
    Y_for_test = []
    for pair in data_for_test:
        X_for_test.append(pair[0])
        Y_for_test.append([pair[1]])
    
    # Train list was generated

    # if validation_data_part != 0:
    #     full_words_array_for_validation = np.concatenate((parts_for_validation[1], parts_for_validation[0]))
    #     labels_for_validation = np.concatenate((np.zeros(len(parts_for_validation[1])), np.ones(len(parts_for_validation[0]))))
    #
    #     data_for_validation = []
    #     for x, y in zip(full_words_array_for_validation, labels_for_validation):
    #         data_for_validation.append([x, y])
    #     random.shuffle(data_for_validation)
    #
    #     X_for_validation = []
    #     Y_for_validation = []
    #     for pair in data_for_validation:
    #         X_for_validation.append(pair[0])
    #         Y_for_validation.append([pair[1]])
        
    # Validation list was generated
    
    finish = time.perf_counter()
    print("Dataset was generated in {} seconds time".format(finish - start))

    train_data = zip(tensor(get_words_embeddings(X_for_train, model)), tensor(Y_for_train))
    test_data = zip(tensor(get_words_embeddings(X_for_test, model)), tensor(Y_for_test))

    # if validation_data_part != 0:
    #     validation_data = zip(tensor(get_words_embeddings(X_for_validation, model)), tensor(Y_for_validation))

    train_data_size = int(len(X_for_train))
    test_data_size = int(len(X_for_test))
    # validation_data_size = int(len(X_for_validation))

    if train_data_part == 0:
        train_dataLoader = None
    else:
        train_dataLoader = DataLoader(list(train_data), batch_size=batch_size, shuffle=True)

    test_dataLoader = DataLoader(list(test_data), batch_size=batch_size, shuffle=True)

    if validation_data_part != 0:
        parts_for_validation = parts_for_test = [
            thematic_words_list[int(len(thematic_words_list) * (1 - validation_data_part)):],
            casual_words_list[int(len(casual_words_list) * (1 - validation_data_part)):]]

        full_words_array_for_validation = np.concatenate((parts_for_validation[1], parts_for_validation[0]))
        labels_for_validation = np.concatenate(
            (np.zeros(len(parts_for_validation[1])), np.ones(len(parts_for_validation[0]))))

        data_for_validation = []
        for x, y in zip(full_words_array_for_validation, labels_for_validation):
            data_for_validation.append([x, y])
        random.shuffle(data_for_validation)

        X_for_validation = []
        Y_for_validation = []
        for pair in data_for_validation:
            X_for_validation.append(pair[0])
            Y_for_validation.append([pair[1]])

        validation_data_size = int(len(X_for_validation))

        validation_data = zip(tensor(get_words_embeddings(X_for_validation, model)), tensor(Y_for_validation))
        validation_dataLoader = DataLoader(list(validation_data), batch_size=batch_size, shuffle=True)

        return train_dataLoader, test_dataLoader, validation_dataLoader, train_data_size, test_data_size, validation_data_size

    # return torch.tensor(get_words_embeddings(X, model)), torch.tensor(Y)
    return train_dataLoader, test_dataLoader, train_data_size, test_data_size


def generate_sets_from_words_for_autoencoder(thematic_words_list, casual_words_list, batch_size, model=None,
                                             delete_casual_words=False, train_data_part=0.8):
    start = time.perf_counter()
    print("Set generation was started")

    if delete_casual_words:
        casual_words_list = []

    parts_for_train = [thematic_words_list[:int(len(thematic_words_list) * train_data_part)],
                       casual_words_list[:int(len(casual_words_list) * train_data_part)]]
    parts_for_test = [thematic_words_list[int(len(thematic_words_list) * train_data_part):],
                      casual_words_list[int(len(casual_words_list) * train_data_part):]]

    full_words_array_for_train = np.concatenate((parts_for_train[1], parts_for_train[0]))
    # labels_for_train = np.concatenate((np.zeros(len(parts_for_train[1])), np.ones(len(parts_for_train[0]))))
    if not delete_casual_words:
        for i in range(7):
            full_words_array_for_train = np.concatenate((full_words_array_for_train, parts_for_train[0]))
            # labels_for_train = np.concatenate((labels_for_train, np.ones(len(parts_for_train[0]))))
    labels_for_train = full_words_array_for_train.copy()

    data_for_train = []
    for x, y in zip(full_words_array_for_train, labels_for_train):
        data_for_train.append([x, y])
    random.shuffle(data_for_train)

    X_for_train = []
    # Y_for_train = []
    for pair in data_for_train:
        X_for_train.append(pair[0])
        # Y_for_train.append([pair[1]])

    # Test list was generated

    full_words_array_for_test = np.concatenate((parts_for_test[1], parts_for_test[0]))
    # labels_for_test = np.concatenate((np.zeros(len(parts_for_test[1])), np.ones(len(parts_for_test[0]))))
    labels_for_test = full_words_array_for_test.copy()

    data_for_test = []
    for x, y in zip(full_words_array_for_test, labels_for_test):
        data_for_test.append([x, y])
    random.shuffle(data_for_test)

    X_for_test = []
    # Y_for_test = []
    for pair in data_for_test:
        X_for_test.append(pair[0])
        # Y_for_test.append(pair[1])

    finish = time.perf_counter()
    print("Dataset was generated in {} seconds time".format(finish - start))

    # print(X_for_train)
    # print(Y_for_train)
    X_for_train = tensor(get_words_embeddings(X_for_train, model))
    X_for_test = tensor(get_words_embeddings(X_for_test, model))

    print("\nTrain data size = " + str(len(X_for_train)) + "\nTest data size = " + str(len(X_for_test)))

    train_data = zip(X_for_train, X_for_train)
    test_data = zip(X_for_test, X_for_test)

    train_data_size = int(len(X_for_train))
    test_data_size = int(len(X_for_test))

    train_dataLoader = DataLoader(list(train_data), batch_size=batch_size, shuffle=True)
    test_dataLoader = DataLoader(list(test_data), batch_size=batch_size, shuffle=True)

    # return torch.tensor(get_words_embeddings(X, model)), torch.tensor(Y)
    return train_dataLoader, test_dataLoader, train_data_size, test_data_size


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


if __name__ == '__main__':
    batch_size = 32
    model_FastTest = fasttext.load_model('../model/cc.en.300.bin')

    thematic_words, casual_words = work_with_files.import_words(load_all_words=True)
    print("Number of thematic words =", len(thematic_words), "Number of casual words =", len(casual_words))
    train_dataLoader, test_dataLoader, validation_dataLoader, train_data_size, test_data_size, validation_data_size = generate_sets_from_words(thematic_words,
                                                                                                  casual_words,
                                                                                                  batch_size,
                                                                                                  model_FastTest, validation_data_part=0.1, train_data_part=0.6)

    print(train_data_size, test_data_size, validation_data_size)

    for x, y in train_dataLoader:
        print(x, y)
        break
    # train_data = zip(X[:int(len(X) * 0.8)], Y[:int(len(X) * 0.8)])
    # test_data = zip(X[int(len(X) * 0.8):], Y[int(len(X) * 0.8):])
    # 
    # train_data_size = int(len(X) * 0.8)
    # test_data_size = len(X) - int(len(X) * 0.8)
    # 
    # train_dataLoader = DataLoader(list(train_data), batch_size=batch_size, shuffle=True)
    # test_dataLoader = DataLoader(list(test_data), batch_size=batch_size, shuffle=True)
