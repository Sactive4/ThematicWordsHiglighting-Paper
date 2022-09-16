import time
import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
import engine
import multiprocessing


def import_words(load_all_words = False):
    start = time.perf_counter()
    "Data import was started!"
    import json

    with open('science_folder/material.json') as file:
        thematic_words_list = json.load(file)['thematic_words']
    random.shuffle(thematic_words_list)

    train_size = int(len(thematic_words_list) * 0.3)
    # thematic_words_train = thematic_words[:train_size]
    # thematic_words_test = thematic_words[train_size:]

    with open('science_folder/casual.json') as file:
        casual_words_list = json.load(file)['casual']
    random.shuffle(casual_words_list)

    thematic_words_list = np.asarray(thematic_words_list)
    casual_words_list = np.asarray(casual_words_list)

    if load_all_words:
        with open('science_folder/new_not_thematic_words_1.1.2.json') as file:
            new_not_thematic_words = np.asarray(json.load(file)['new_not_thematic_words'])
            new_list = np.append(new_not_thematic_words, casual_words_list)
            casual_words_list = np.unique(new_list)
            random.shuffle(casual_words_list)
        
        with open('science_folder/new_thematic_words_1.1.2.json') as file:
            new_thematic_words = np.asarray(json.load(file)['new_thematic_words'])
            new_list = np.append(new_thematic_words, thematic_words_list)
            thematic_words_list = np.unique(new_list)
            random.shuffle(thematic_words_list)

    finish = time.perf_counter()
    print("Data import FINISHED in {} seconds time".format(finish - start))
    return thematic_words_list, casual_words_list


if __name__ == '__main__':
    import fasttext

    # model = fasttext.load_model('../model/cc.en.300.bin')
    start = time.perf_counter()
    print("Work_with_files was started")

    thematic_words, casual_words = import_words(load_all_words=True)
    print(len(thematic_words), len(casual_words))

    # X, Y = engine.generate_sets_from_words(thematic_words, casual_words, model)
    # print(X)
    # print(Y)

    finish = time.perf_counter()
    print("Full process FINISHED in {} seconds time".format(finish - start))
