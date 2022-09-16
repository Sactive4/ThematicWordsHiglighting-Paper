import shutil
import sys
import warnings
import fasttext
import time
import pandas as pd
import numpy as np
import os
import torch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch import tensor
import torchvision
import torchvision.transforms as transforms

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import work_with_files
import engine


class IdeenPerceptron(nn.Module):

    def __init__(self, input=300, h1=20, h2=None):
        super(IdeenPerceptron, self).__init__()
        self.device = torch.device("cpu")

        if h2 is None:
            self.model = nn.Sequential(
                nn.Linear(input, h1),
                nn.Sigmoid(),
                nn.Linear(h1, 1),
                nn.Sigmoid()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input, h1),
                nn.Sigmoid(),
                nn.Linear(h1, h2),
                nn.Sigmoid(),
                nn.Linear(h2, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # x = torch.flatten(x, 1)
        return self.model(x)

    def train_itself(self, train_dataLoader, epochs=10, lr=0.01, device_name="cpu"):
        # start = time.perf_counter()

        self.device = torch.device(device_name)
        self.to(self.device)

        # if model_FastTest is None:
        #     model_FastTest = fasttext.load_model('../model/cc.en.300.bin')
        #
        # thematic_words, casual_words = work_with_files.import_words(load_all_words=True)
        # print("Number of thematic words =", len(thematic_words), "Number of casual words =", len(casual_words))
        # train_dataLoader, test_dataLoader, train_data_size, test_data_size = engine.generate_sets_from_words_for_autoencoder(
        #     thematic_words_list=thematic_words, casual_words_list=casual_words, batch_size=batch_size,
        #     model=model_FastTest, delete_casual_words=False)

        #

        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        transform = transforms.ToTensor()

        # start_learning = time.perf_counter()
        # print("Learning was started!")

        # accuracy_array = []

        for epoch in range(epochs):
            # print("epoch № " + str(epoch) + "\n" + str("_" * 100))

            train_loss = 0

            for embedding, labels in train_dataLoader:
                optimizer.zero_grad()

                embedding = embedding.to(self.device)
                labels = labels.to(self.device)
                output = self(embedding)
                # print(output, labels)
                # print(loss_function(output, labels))

                loss = loss_function(output, labels)
                # loss = nn.L1Loss(output, labels)
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            # test_loss = 0
            # correct = 0
            # true_positives = 0
            # false_and_true_positives = 0
            # relevant_items = 0

            # for embedding, labels in tqdm(test_dataLoader):
            #     # print(embedding, labels)
            #     with torch.no_grad():
            #         embedding = embedding.to(device)
            #         labels = labels.to(device)
            #         output = self(embedding)
            #         if device_name == "cuda:0":
            #             predictions = tensor(output.cpu().numpy().round(2)).to(device)
            #         else:
            #             predictions = tensor(output.numpy().round(2))
            #         # print(output, labels)
            #         loss = loss_function(output, labels)
            #         # true_positives += (predictions[predictions == labels] == 1).sum()
            #         # false_and_true_positives += predictions.sum()
            #         # relevant_items += labels.sum()
            #         test_loss += loss.item()
            #         if device_name == "cuda:0":
            #             correct += (predictions == tensor(labels.cpu().numpy().round(2)).to(device)).sum()
            #         else:
            #             correct += (predictions == tensor(labels.numpy().round(2))).sum()
            #
            # print(correct)
            # print(f"Train loss * 1000: {1000 * train_loss / train_data_size:.3f}")
            # print(f"Test loss * 1000: {1000 * test_loss / test_data_size:.3f}")
            # print(f"Test accuracy: {correct / test_data_size / 300:.3f}")
            # print(f"Test Recall: {float(true_positives / relevant_items) :.3f}")
            # print(f"Test Precision: {float(true_positives / false_and_true_positives) :.3f}")
            # print(f"Test F1 SCORE: {np.sqrt(float(true_positives**2 / relevant_items / false_and_true_positives)) :.3f}")

            # if epoch >= 3:
            #     accuracy_array.append(round(float(correct / test_data_size / 300), 4))
            #     if accuracy_array[-1] >= 0.999:
            #         epochs = epoch + 1
            #         break

        # finish_learning = time.perf_counter()
        # print("Learning was finished in {} seconds time".format(finish_learning - start_learning))

        # number_of_layers = 1
        # if self.h2 is not None:
        #     number_of_layers = 2
        # self.draw_embeddings_graphs_and_save_embeddings(number_of_layers=number_of_layers, model_FastTest=model_FastTest,
        #                                                 device_name=device_name, load_all_words=True)

        # fig, ax = plt.subplots()
        # plt.plot(list(range(3, epochs)), accuracy_array)
        # plt.savefig("NN_accuracy.jpg")

        # print("MAX accuracy = " + str(max(accuracy_array)))
        #
        # device = torch.device("cpu")
        # self.to(device)

        # torch.save(self, "../model/IdeenNN-V4-" + str(self.h1) + "-with-" + str(
        #     round(max(accuracy_array), 3)) + "-accuracy.pth")

        # finish = time.perf_counter()
        # print("Learning and drawing embeddings FINISHED in {} seconds time".format(finish - start))

    def predict_proba(self, X):
        with torch.no_grad():
            X = tensor(X).to(self.device)
            Y = self(X).cpu().numpy()[:, 0]
            return np.array(list((zip(np.ones_like(Y) - Y, Y))))

    def save(self, path="../model/IdeenPerceptron-NoV"):
        device = torch.device("cpu")
        self.to(device)

        torch.save(self, path)


class IdeenPerceptronForGettingEmbedding(nn.Module):

    def __init__(self, custom_weight, number_of_layers=1):
        super(IdeenPerceptronForGettingEmbedding, self).__init__()

        layers = []
        for index in range(number_of_layers):
            layers.append(torch.nn.Linear(300, 7))
            layers[-1].weight = nn.Parameter(custom_weight[2 * index])
            layers[-1].bias = nn.Parameter(custom_weight[2 * index + 1])

        # layer = torch.nn.Linear(300, 7)
        # layer.weight = nn.Parameter(custom_weight[0])
        # layer.bias = nn.Parameter(custom_weight[1])

        if number_of_layers == 1:
            self.model = nn.Sequential(
                layers[0],
                nn.Sigmoid()
            )
        elif number_of_layers == 2:
            self.model = nn.Sequential(
                layers[0],
                nn.Sigmoid(),
                layers[1],
                nn.Sigmoid()
            )
        elif number_of_layers == 3:
            self.model = nn.Sequential(
                layers[0],
                nn.Sigmoid(),
                layers[1],
                nn.Sigmoid(),
                layers[2],
                nn.Sigmoid()
            )

        # self.embeddings = None
        # self.compress_embeddings = None
        # self.thematic_embeddings = None

    def forward(self, x):
        if x.shape != (len(x), 300):
            raise TypeError("x.shape is not 2-d array with 300-d embeddings")
        return self.model(x)


class IdeenAutoEncoder(nn.Module):

    def __init__(self, input=300, h1=20, h2=None):
        super(IdeenAutoEncoder, self).__init__()
        self.h1 = h1
        self.input = input
        self.h2 = h2
        self.all_casual_embeddings = None
        self.all_thematic_embeddings = None
        self.embeddings = None
        self.compress_embeddings = None
        self.thematic_embeddings = None
        if h2 is None:
            self.model = nn.Sequential(
                nn.Linear(input, h1),
                nn.Sigmoid(),
                nn.Linear(h1, input)
                # nn.Sigmoid()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input, h1),
                nn.Sigmoid(),
                nn.Linear(h1, h2),
                nn.Sigmoid(),
                nn.Linear(h2, h1),
                nn.Sigmoid(),
                nn.Linear(h1, input)
                # nn.Sigmoid()
            )

    def forward(self, x):
        return self.model(x)

    def draw_embeddings_graphs_and_save_embeddings(self, device_name="cpu", number_of_layers=1, model_FastTest=None,
                                                   load_all_words=False):
        import work_with_files
        import random
        import engine

        start = time.perf_counter()
        print("Draw_embeddings_graphs was started!")

        device = torch.device(device_name)
        # self = torch.load("../model/IdeenPerceptron-V3-7-with-0.901-F1-and.pth")
        if model_FastTest is None:
            model_FastTest = fasttext.load_model('../model/cc.en.300.bin')
        model_FastTest7 = fasttext.load_model('../model/cc.en.7.bin')

        print("Models were imported!")

        thematic_words, casual_words = work_with_files.import_words(load_all_words=load_all_words)
        random.shuffle(thematic_words)
        random.shuffle(casual_words)

        words_list = []
        words_list.extend(casual_words[:5])
        words_list.extend(thematic_words[:5])
        thematic_embeddings_size = 20

        parameters = []
        for param in self.parameters():
            parameters.append(param.type(torch.DoubleTensor).to(device))
        # print(self.parameters)
        # print(parameters)

        # if layer_number % 2 == 1:
        #     raise KeyError("layer_number = " + str(layer_number) + " that means you will receive prams from 2 different layers, did you mean " + str(layer_number-1) + "?")

        net_for_getting_embeddings = IdeenPerceptronForGettingEmbedding(
            custom_weight=parameters, number_of_layers=number_of_layers)

        self.embeddings = engine.get_words_embeddings(words_list, model_FastTest)
        self.all_casual_embeddings = net_for_getting_embeddings(
            tensor(engine.get_words_embeddings(casual_words, model_FastTest), dtype=torch.double).to(device)).cpu()
        self.all_thematic_embeddings = net_for_getting_embeddings(
            tensor(engine.get_words_embeddings(thematic_words, model_FastTest), dtype=torch.double).to(device)).cpu()
        embeddings7 = engine.get_words_embeddings(words_list, model_FastTest7)
        self.compress_embeddings = net_for_getting_embeddings(
            tensor(self.embeddings, dtype=torch.double).to(device)).cpu()
        self.thematic_embeddings = net_for_getting_embeddings(
            tensor(engine.get_words_embeddings(thematic_words[:thematic_embeddings_size], model_FastTest),
                   dtype=torch.double).to(device)).cpu()

        if os.path.exists("data/compressed_in_autoencoder_embeddings"):
            shutil.rmtree("data/compressed_in_autoencoder_embeddings")
        os.makedirs("data/compressed_in_autoencoder_embeddings")

        fig, ax = plt.subplots()
        ax.imshow(self.embeddings)
        plt.yticks(list(range(len(words_list))), words_list)
        plt.savefig("data/" + "compressed_in_autoencoder_embeddings/embeddings.jpg")

        fig, ax = plt.subplots()
        ax.imshow(embeddings7)
        plt.yticks(list(range(len(words_list))), words_list)
        plt.savefig("data/" + "compressed_in_autoencoder_embeddings/embeddings7.jpg")

        fig, ax = plt.subplots()
        ax.imshow(self.compress_embeddings.detach().numpy())
        plt.yticks(list(range(len(words_list))), words_list)
        plt.savefig("data/" + "compressed_in_autoencoder_embeddings/compressed_in_autoencoder_embeddings.jpg")

        fig, ax = plt.subplots()
        ax.imshow(self.thematic_embeddings.detach().numpy())
        plt.yticks(list(range(thematic_embeddings_size)), thematic_words[:thematic_embeddings_size])
        plt.savefig("data/" + "compressed_in_autoencoder_embeddings/compressed_in_autoencoder_thematic_embeddings.jpg")

        finish = time.perf_counter()
        print("Draw_embeddings_graphs was FINISHED in {} seconds time".format(finish - start))

    def train_itself(self, model_FastTest=None, epochs=10, batch_size=32, lr=0.01, device_name="cpu",
                     delete_casual_words=False, load_all_words=False):
        start = time.perf_counter()

        device = torch.device(device_name)
        self.to(device)
        if model_FastTest is None:
            model_FastTest = fasttext.load_model('../model/cc.en.300.bin')

        thematic_words, casual_words = work_with_files.import_words(load_all_words=True)
        print("Number of thematic words =", len(thematic_words), "Number of casual words =", len(casual_words))
        train_dataLoader, test_dataLoader, train_data_size, test_data_size = engine.generate_sets_from_words_for_autoencoder(
            thematic_words_list=thematic_words, casual_words_list=casual_words, batch_size=batch_size,
            model=model_FastTest, delete_casual_words=delete_casual_words)

        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        transform = transforms.ToTensor()

        start_learning = time.perf_counter()
        print("Learning was started!")

        accuracy_array = []

        for epoch in tqdm(range(epochs)):
            # print("epoch № " + str(epoch) + "\n" + str("_" * 100))

            train_loss = 0

            for embedding, labels in train_dataLoader:
                optimizer.zero_grad()

                embedding = embedding.to(device)
                labels = labels.to(device)
                output = self(embedding)

                loss = loss_function(output, labels)
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            test_loss = 0
            correct = 0
            # true_positives = 0
            # false_and_true_positives = 0
            # relevant_items = 0

            for embedding, labels in test_dataLoader:
                with torch.no_grad():
                    embedding = embedding.to(device)
                    labels = labels.to(device)
                    output = self(embedding)
                    if device_name == "cuda:0":
                        predictions = tensor(output.cpu().numpy().round(2)).to(device)
                    else:
                        predictions = tensor(output.numpy().round(2))
                    loss = loss_function(output, labels)
                    test_loss += loss.item()
                    if device_name == "cuda:0":
                        correct += (predictions == tensor(labels.cpu().numpy().round(2)).to(device)).sum()
                    else:
                        correct += (predictions == tensor(labels.numpy().round(2))).sum()

            # print(correct)
            # print(f"Train loss * 1000: {1000 * train_loss / train_data_size:.3f}")
            # print(f"Test loss * 1000: {1000 * test_loss / test_data_size:.3f}")
            # print(f"Test accuracy: {correct / test_data_size / 300:.3f}")
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

        number_of_layers = 1
        if self.h2 is not None:
            number_of_layers = 2
        self.draw_embeddings_graphs_and_save_embeddings(number_of_layers=number_of_layers,
                                                        model_FastTest=model_FastTest,
                                                        device_name=device_name, load_all_words=load_all_words)

        fig, ax = plt.subplots()
        plt.plot(list(range(3, epochs)), accuracy_array)
        plt.savefig("data/" + "autoencoder_accuracy.jpg")

        fig, axis = plt.subplots(figsize=(10, 10))
        plt.scatter(self.all_thematic_embeddings.detach().numpy()[:, 0],
                    self.all_thematic_embeddings.detach().numpy()[:, 1],
                    color="r",
                    marker='^')
        plt.savefig(
            "data/" + "compressed_in_autoencoder_embeddings/2D thematic embeddings from Autoencoder visualization 2.0 V Test8.png",
            dpi=300)

        print("MAX accuracy = " + str(max(accuracy_array)))

        device = torch.device("cpu")
        self.to(device)

        torch.save(self, "../model/IdeenAutoEncoder-V1-" + str(self.h1) + "-with-" + str(
            round(max(accuracy_array), 3)) + "-accuracy.pth")

        finish = time.perf_counter()
        print("Learning and drawing embeddings FINISHED in {} seconds time".format(finish - start))


class IdeenModelTrainer:

    def __init__(self, working_dir="IdeenModelTrainer_working_directory", version="V-1.1.1", comment="",
                 model_FastTest_loaded=None,
                 load_all_words=False, batch_size=16,
                 repetition_of_thematic_selection=0, train_data_part=0.6, validation_data_part=0.2):

        self.version = version
        self.comment = comment
        if model_FastTest_loaded is None:
            self.model_FastTest = fasttext.load_model('../model/cc.en.300.bin')
        else:
            self.model_FastTest = model_FastTest_loaded

        self.dir_name = "data/" + working_dir + "-" + version

        if os.path.exists(self.dir_name):
            shutil.rmtree(self.dir_name)
        os.makedirs(self.dir_name)

        sys.stdout = open(self.dir_name + "/output.txt", 'w')

        thematic_words, casual_words = work_with_files.import_words(load_all_words=load_all_words)
        print(len(thematic_words), len(casual_words))
        self.train_dataLoader, self.test_dataLoader, self.validation_dataLoader, self.train_data_size, self.test_data_size, self.validation_data_size = engine.generate_sets_from_words(
            thematic_words_list=thematic_words, casual_words_list=casual_words, batch_size=batch_size,
            model=self.model_FastTest,
            repetition_of_thematic_selection=repetition_of_thematic_selection, train_data_part=train_data_part,
            validation_data_part=validation_data_part)

        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        X_valid = []
        Y_valid = []

        for embedding, labels in self.train_dataLoader:
            X_train.extend(list(embedding.numpy()))
            Y_train.extend(list(labels.numpy()))
        for embedding, labels in self.test_dataLoader:
            X_test.extend(list(embedding.numpy()))
            Y_test.extend(list(labels.numpy()))
        for embedding, labels in self.validation_dataLoader:
            X_valid.extend(list(embedding.numpy()))
            Y_valid.extend(list(labels.numpy()))

        self.X_valid = np.asarray(X_valid)
        self.Y_valid = np.asarray(Y_valid)[:, 0]
        self.X_test = np.asarray(X_test)
        self.X_train = np.asarray(X_train)
        self.Y_test = np.asarray(Y_test)[:, 0]
        self.Y_train = np.asarray(Y_train)[:, 0]
        # print(self.X_train)
        # print(self.Y_train)

    def train_model(self, train_func, params_names, params_values,
                    number_of_steps=20, repeats=1, save_model=False, save_pdf_graphs=True):

        start_learning = time.perf_counter()
        print("Learning was started!")
        time.sleep(0.001)

        threshold_array = np.asarray(list(range(number_of_steps + 1))) / number_of_steps

        best_params = (None, None)
        best_accuracy = 0
        best_F1_score = 0
        best_threshold = -1

        if len(params_names) != len(params_values):
            raise ValueError("length of params_names != length of params_values")
        if len(np.shape(params_values)) != 2:
            # print(np.shape(params_values))
            warnings.warn("params_values could be not 2D array!")
            time.sleep(0.001)

        def generate_params_grid(params_names_for_gen, params_values_for_gen, repeats_for_gen,
                                 params_for_gen=()):

            if len(params_values_for_gen) == 0:
                if len(params_for_gen) == 0:
                    raise ValueError("There is no parameters for generation grids")
                return params_for_gen * repeats_for_gen

            if len(params_for_gen) == 0:
                parameter_values = [[param_value] for param_value in params_values_for_gen[0]]
                return generate_params_grid(params_names_for_gen[1:], params_values_for_gen[1:],
                                            params_for_gen=parameter_values, repeats_for_gen=repeats_for_gen)

            new_params_for_gen = []
            for params_grid_index in range(len(params_for_gen)):
                new_params_for_gen.extend([params_for_gen[params_grid_index] + [new_param_value] for new_param_value in
                                           params_values_for_gen[0]])
            return generate_params_grid(params_names_for_gen[1:], params_values_for_gen[1:],
                                        params_for_gen=new_params_for_gen, repeats_for_gen=repeats_for_gen)

        params_array = generate_params_grid(params_names_for_gen=params_names, params_values_for_gen=params_values,
                                            repeats_for_gen=repeats)

        for params in tqdm(params_array):
            model = train_func(params=params, train_dataLoader=self.train_dataLoader)
            if model is None:
                continue
            precision = []
            recall = []
            accuracy = []
            F1_score = []

            output = model.predict_proba(self.X_test)[:, 1]
            for threshold in threshold_array:  # Computing F1 for different threshold
                prediction = engine.round_nd_array(output, threshold)
                true_positives = sum(self.Y_test[self.Y_test == prediction] == 1)
                false_and_true_positives = sum(prediction)
                relevant_items = sum(self.Y_test)
                correct = sum((self.Y_test == prediction))

                accuracy.append(correct / len(self.Y_test))
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
                best_params = params
                # best_threshold = threshold

        print("Best F1 =\t" + str(best_F1_score))
        # print("Best Params: \t\t" + params_names[0] + " = " + str(best_params[0]) + "\t\t" + params_names[1] + " = " + str(best_params[1]) + "\t\t" + params_names[2] + " = " + str(best_params[2]) + "\t\tepoch = " + str(epoch))
        print("Best Params:")
        for param_index in range(len(best_params)):
            print("\t\t" + params_names[param_index] + " = " + str(best_params[param_index]), end="")
        print("\nBest accuracy =\t" + str(best_accuracy))

        # model_net = IdeenPerceptron(input=300, h1=h1)
        # model_net.train_itself(train_dataLoader=train_dataLoader, epochs=epoch, lr=lr)
        model = train_func(params=best_params, train_dataLoader=self.train_dataLoader)

        time.sleep(0.001)

        precision = []
        recall = []
        accuracy = []
        F1_score = []
        relevant_items = sum(self.Y_valid)

        number_of_steps = 500
        threshold_array = np.asarray(list(range(number_of_steps + 1))) / number_of_steps

        output = model.predict_proba(self.X_valid)

        for threshold in tqdm(threshold_array):  # Computing F1 for different threshold
            prediction = engine.round_nd_array(output, threshold)[:, 1]
            # prediction = engine.round_nd_array(clf.predict_proba(X_valid)[:, 1], threshold)
            true_positives = sum(self.Y_valid[self.Y_valid == prediction] == 1)
            false_and_true_positives = sum(prediction)

            correct = sum((self.Y_valid == prediction))
            accuracy.append(correct / len(self.Y_valid))
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

        if save_pdf_graphs:
            plt.savefig(self.dir_name + "/F1_score_function of a model" + self.version + ".pdf", dpi=300)
        else:
            plt.savefig(self.dir_name + "/F1_score_function of a model " + self.version + ".jpg", dpi=300)

        if save_model:
            model.save(
                "../model/IdeenPerceptron-" + self.version + "-with-" + str(
                    max(F1_score).round(3)) + "-F1-and.pth")
            model.save(
                self.dir_name + "/IdeenPerceptron-" + self.version + "-" + self.comment + "-with-" + str(
                    max(F1_score).round(3)) + "-F1-and.pth")

        finish_learning = time.perf_counter()
        print("Learning was finished in {} seconds time".format(finish_learning - start_learning))


def train_function_NN(params, train_dataLoader):
    # params_values=[[300], [3, 5, 7, 10], [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075], [3, 5, 7, 10, 15, 20, 25, 35]]
    # params_names=["input", "h1", "lr", "epoch"]

    # train_func = train_function_NN,
    # params_names = ["input", "h1", "h2", "lr", "epoch"],
    # params_values = [[300], [3, 5, 7, 10, 15, 20, 25, 50, 100], [None],
    #                  [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
    #                  [3, 5, 7, 10, 15, 20, 25, 35, 50]],
    # repeats = 3, save_model = False, save_pdf_graphs = False

    if params[2] is None or params[1] < 2.5 * params[2]:
        return None

    model = IdeenPerceptron(input=params[0], h1=params[1], h2=params[2])
    model.train_itself(train_dataLoader=train_dataLoader, epochs=params[4], lr=params[3])
    return model


def train_function_SVM(params, train_dataLoader):
    # c_array = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5]
    # kernel_array = ['linear', 'poly', 'rbf', 'sigmoid']

    # train_func = train_function_SVM,
    # number_of_params_for_init = 2, params_names = ["kernel", "c"],
    # params_values = [['rbf'],
    #                  [2.5]],
    # repeats = 3, save_model = False, save_pdf_graphs = False

    X_train = []
    Y_train = []
    for embedding, labels in train_dataLoader:
        X_train.extend(list(embedding.numpy()))
        Y_train.extend(list(labels.numpy()))
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)[:, 0]

    clf = svm.SVC(kernel=params[0], cache_size=1000, C=params[1], probability=True)
    clf.fit(X_train, Y_train)
    return clf


def train_function_RandomForest(params, train_dataLoader):
    # n_estimator = [10, 20, 30, 50, 75, 100, 150]
    # criterion = ["gini", "entropy", "log_loss"]
    # max_depth = list(range(3, 40))

    # train_func = train_function_RandomForest,
    # number_of_params_for_init = 3, params_names = ["n_estimator", "criterion", "max_depth"],
    # params_values = [[10, 20, 30, 50, 75, 100, 150], ["gini", "entropy", "log_loss"], list(range(3, 10))],
    # repeats = 3, save_model = False, save_pdf_graphs = False

    X_train = []
    Y_train = []
    for embedding, labels in train_dataLoader:
        X_train.extend(list(embedding.numpy()))
        Y_train.extend(list(labels.numpy()))
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)[:, 0]

    clf = RandomForestClassifier(n_estimators=params[0], max_depth=params[2], criterion=params[1], random_state=None,
                                 n_jobs=-1)
    clf.fit(X_train, Y_train)
    return clf


def trainKNN(params, train_dataLoader):
    # weights=['uniform', 'distance'],
    # n_neighbors=list(range(20))

    # train_func = trainKNN,
    # number_of_params_for_init = 2, params_names = ["weights", "n_neighbors"],
    # params_values = [['uniform', 'distance'], list(range(5, 30))],
    # repeats = 3, save_model = False, save_pdf_graphs = False

    from sklearn.neighbors import KNeighborsClassifier

    X_train = []
    Y_train = []
    for embedding, labels in train_dataLoader:
        X_train.extend(list(embedding.numpy()))
        Y_train.extend(list(labels.numpy()))
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)[:, 0]

    neigh = KNeighborsClassifier(n_neighbors=params[1], weights=params[0], n_jobs=7)
    neigh.fit(X_train, Y_train)

    return neigh


if __name__ == '__main__':
    trainer = IdeenModelTrainer(repetition_of_thematic_selection=3, working_dir="NN_classifier",
                                version="V_TEST-1.1.3", load_all_words=False)
    trainer.train_model(train_func=train_function_NN,
                        params_names=["input", "h1", "h2", "lr", "epoch"],
                        params_values=[[300], [10, 20, 50, 100], [3, 5, 7, 10, 15],
                                       [0.002, 0.005, 0.01, 0.02, 0.03, 0.05],
                                       [3, 5, 7, 10, 15, 20, 25, 35, 50]],
                        repeats=1, save_model=False, save_pdf_graphs=False)
