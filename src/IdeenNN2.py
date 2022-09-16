import os
import shutil

from sklearn import svm
import work_with_files
import engine
import fasttext
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
from classifier import IdeenPerceptron
import sys


if __name__ == '__main__':

    # device_name = "cuda:0"
    device_name = "cpu"
    version = "Vh2-2.-1.3"

    working_dir = "data/" + "working_with_NN_classifier-" + version

    if os.path.exists(working_dir):
        print("This directory is already exists, do you want to rewrite this directory?\nY / N")
        if input() == "Y":
            shutil.rmtree(working_dir)
        else:
            exit(1)
    os.makedirs(working_dir)

    print("Directory was rewritten")

    sys.stdout = open(working_dir + "/output.txt", 'w')

    model_FastTest = fasttext.load_model('../model/cc.en.300.bin')
    thematic_words, casual_words = work_with_files.import_words(load_all_words=True)
    train_dataLoader, test_dataLoader, validation_dataLoader, train_data_size, test_data_size, validation_data_size = engine.generate_sets_from_words(
        thematic_words_list=thematic_words, casual_words_list=casual_words, batch_size=256, model=model_FastTest,
        repetition_of_thematic_selection=5, train_data_part=0.6, validation_data_part=0.2)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_valid = []
    Y_valid = []

    for embedding, labels in tqdm(train_dataLoader):
        X_train.extend(list(embedding.numpy()))
        Y_train.extend(list(labels.numpy()))
    for embedding, labels in tqdm(test_dataLoader):
        X_test.extend(list(embedding.numpy()))
        Y_test.extend(list(labels.numpy()))
    for embedding, labels in tqdm(validation_dataLoader):
        X_valid.extend(list(embedding.numpy()))
        Y_valid.extend(list(labels.numpy()))

    X_valid = np.asarray(X_valid)
    Y_valid = np.asarray(Y_valid)[:, 0]
    X_test = np.asarray(X_test)
    X_train = np.asarray(X_train)
    Y_test = np.asarray(Y_test)[:, 0]
    Y_train = np.asarray(Y_train)[:, 0]

    start_learning = time.perf_counter()
    print("Learning was started!")
    time.sleep(0.001)

    number_of_steps = 20
    threshold_array = np.asarray(list(range(number_of_steps + 1))) / number_of_steps

    best_params = (None, None)
    best_accuracy = 0
    best_F1_score = 0
    best_threshold = -1

    params = []

    h1_array = [10, 15, 20, 25, 50]
    # h1_array = [3, 5, 7, 10, 15, 20, 25, 50, 100, 300, 500]
    # h1_array = [300, 500]
    # h1_array = [20, 50, 100]
    # h1_array = [3, 5, 7, 10]
    h2_array = [3, 5, 7, 10]
    # h2_array = [3, 5, 7, 10, 15, 20, 25, 50]
    # h2_array = [10, 25]
    # h2_array = [5, 7, 10, 15]
    # h2_array = [None]
    # lr_array = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
    lr_array = [0.002, 0.005, 0.01, 0.025]
    # lr_array = [0.002, 0.005, 0.01]
    # lr_array = [0.005, 0.01]
    # epoch_array = [3, 5, 7, 10, 15, 20, 25, 35]
    # epoch_array = [10, 15, 20, 25, 35, 50, 100, 200]
    epoch_array = [10, 15, 25, 50]
    # epoch_array = [50, 100, 200]
    # epoch_array = [10, 15, 20, 25, 35, 50, 100]

    repeats = 1
    for h1 in h1_array:
        for h2 in h2_array:
            if h2 is not None and h2 * 2 > h1:
                continue
            for lr in lr_array:
                for epoch in epoch_array:
                    for repeat in range(repeats):
                        params.append((h1, h2, lr, epoch))

    for h1, h2, lr, epoch in tqdm(params):

        model_net = IdeenPerceptron(input=300, h1=h1, h2=h2)
        model_net.train_itself(train_dataLoader=train_dataLoader, epochs=epoch, lr=lr, device_name=device_name)

        precision = []
        recall = []
        accuracy = []
        F1_score = []

        output = model_net.predict_proba(X_test)[:, 1]
        for threshold in threshold_array:  # Computing F1 for different threshold
            prediction = engine.round_nd_array(output, threshold)
            true_positives = sum(Y_test[Y_test == prediction] == 1)
            false_and_true_positives = sum(prediction)
            relevant_items = sum(Y_test)
            correct = sum((Y_test == prediction))

            accuracy.append(correct / len(Y_test))
            recall.append(float(true_positives / relevant_items))
            if false_and_true_positives == 0:
                precision.append(1.0)
                F1_score.append(0)
                continue
            precision.append(float(true_positives / false_and_true_positives))
            F1_score.append(float(true_positives) / np.sqrt(float(relevant_items * false_and_true_positives)))

        F1_score = max(F1_score)
        accuracy = max(accuracy)

        if F1_score > best_F1_score:
            best_F1_score = F1_score
            best_accuracy = accuracy
            best_params = (h1, h2, lr, epoch)
            # best_threshold = threshold

    h1, h2, lr, epoch = best_params
    
    print("Best F1 =\t" + str(best_F1_score))
    print("Best Params: \t\t" + "h1 = " + str(h1) + "\t\th2 = " + str(h2) + "\t\tlr = " + str(lr) + "\t\tepoch = " + str(epoch))
    print("Best accuracy =\t" + str(best_accuracy))

    model_net = IdeenPerceptron(input=300, h1=h1)
    model_net.train_itself(train_dataLoader=train_dataLoader, epochs=epoch, lr=lr)

    time.sleep(0.001)

    precision = []
    recall = []
    accuracy = []
    F1_score = []
    relevant_items = sum(Y_valid)

    number_of_steps = 500
    threshold_array = np.asarray(list(range(number_of_steps + 1))) / number_of_steps

    for threshold in tqdm(threshold_array):  # Computing F1 for different threshold
        output = model_net.predict_proba(X_valid)
        prediction = engine.round_nd_array(output, threshold)[:, 1]
        # prediction = engine.round_nd_array(clf.predict_proba(X_valid)[:, 1], threshold)
        true_positives = sum(Y_valid[Y_valid == prediction] == 1)
        false_and_true_positives = sum(prediction)

        correct = sum((Y_valid == prediction))
        accuracy.append(correct / len(Y_valid))
        recall.append(float(true_positives / relevant_items))
        if false_and_true_positives == 0:
            precision.append(1.0)
            F1_score.append(0)
        else:
            precision.append(float(true_positives / false_and_true_positives))
            F1_score.append(float(true_positives) / np.sqrt(float(relevant_items * false_and_true_positives)))



    print("MAX F1 = " + str(max(F1_score)))

    fig, axis = plt.subplots(2, figsize=(20, 20))

    axis[0].plot(threshold_array, precision, color='r', label='precision')
    axis[0].plot(threshold_array, recall, color='g', label='recall')
    axis[0].set_xlabel("threshold")
    axis[0].set_ylabel("Magnitude")
    axis[0].set_title("Precision and Recall functions")
    axis[0].legend()
    axis[1].plot(threshold_array, F1_score, color='b', label='F1 score')
    axis[1].plot(threshold_array, accuracy, color='k', label='accuracy')
    axis[1].set_xlabel("threshold")
    axis[1].set_ylabel("Magnitude")
    axis[1].set_title("F1 score function and accuracy")
    axis[1].legend()

    plt.savefig(working_dir + "/F1_score_function of a NN " + version + ".pdf", dpi=300)
    model_net.save(
        "../model/IdeenPerceptron-" + version + "-" + str(h1) + "-with-" + str(max(F1_score).round(3)) + "-F1-and.pth")
    model_net.save(
        working_dir + "/IdeenPerceptron-" + version + "-" + str(h1) + "-with-" + str(max(F1_score).round(3)) + "-F1-and.pth")

    finish_learning = time.perf_counter()
    print("Learning was finished in {} seconds time".format(finish_learning - start_learning))

    # plt.show()
