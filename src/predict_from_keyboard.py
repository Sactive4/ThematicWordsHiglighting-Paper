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
    print("Working_with_model was started!")

    model_net = torch.load("../model/IdeenPerceptron-V3-7-with-0.901-F1-and.pth")
    model_FastTest = fasttext.load_model('../model/cc.en.300.bin')
    print("Models were imported!")

    while True:
        print("enter your word or EXIT")
        word = input()
        if word == "EXIT":
            break
        output = model_net(tensor(engine.get_words_embeddings([word], model_FastTest)))
        prediction = output.round()
        print("word: " + word + "\t\tprediction: " + str(int(prediction)) + "\t\toutput: " + str(output))

    finish = time.perf_counter()
    print("Full process FINISHED in {} seconds time".format(finish - start))

