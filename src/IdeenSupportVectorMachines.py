from sklearn import svm
import work_with_files
import engine
import fasttext
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

if __name__ == '__main__':
    model_FastTest = fasttext.load_model('../model/cc.en.300.bin')
    thematic_words, casual_words = work_with_files.import_words(load_all_words=False)
    train_dataLoader, test_dataLoader, validation_dataLoader, train_data_size, test_data_size, validation_data_size = engine.generate_sets_from_words(
        thematic_words_list=thematic_words, casual_words_list=casual_words, batch_size=16, model=model_FastTest,
        repetition_of_thematic_selection=0, train_data_part=0.6, validation_data_part=0.2)

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

    # print(X_train)
    # print(Y_train)

    start_learning = time.perf_counter()
    print("Learning was started!")
    time.sleep(0.001)

    number_of_steps = 20
    threshold_array = np.asarray(list(range(number_of_steps + 1))) / number_of_steps

    best_params = (None, None)
    best_accuracy = 0
    best_F1_score = 0

    params = []
    # c_array = [0.1, 0.5, 1.0, 2]
    c_array = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5]

    for kernel in tqdm(['linear', 'poly', 'rbf', 'sigmoid']):
        for c in c_array:
            params.append((kernel, c))

    for kernel, c in tqdm(params):

        # clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=None,
        #                              n_jobs=-1)
        clf = svm.SVC(kernel=kernel, cache_size=1000, C=c, probability=True)
        clf.fit(X_train, Y_train)

        precision = []
        recall = []
        accuracy = []
        F1_score = []

        output = clf.predict_proba(X_test)[:, 1]
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

        # prediction = clf.predict(X_test)
        #
        # accuracy = sum((Y_test == prediction)) / len(Y_test)
        # true_positives = sum(Y_test[Y_test == prediction] == 1)
        # false_and_true_positives = sum(prediction)
        # relevant_items = sum(Y_test)
        # if false_and_true_positives == 0:
        #     F1_score = 0
        # else:
        #     F1_score = true_positives / np.sqrt(false_and_true_positives * relevant_items)

        F1_score = max(F1_score)
        accuracy = max(accuracy)

        if F1_score > best_F1_score:
            best_F1_score = F1_score
            best_accuracy = accuracy
            best_params = (kernel, c)

    print("MAX F1 =\t" + str(best_F1_score))
    print("Best Params: \t\t" + "kernel = " + str(best_params[0]) + "\t\tc = " + str(best_params[1]))
    print("Best accuracy =\t" + str(best_accuracy))

    clf = svm.SVC(kernel=best_params[0], cache_size=2500, C=best_params[1],  probability=True)
    clf.fit(X_train, Y_train)

    print("clf.score =\t" + str(clf.score(X_test, Y_test)))
    time.sleep(0.001)
    # probabilities = clf.predict_proba(X_test)

    precision = []
    recall = []
    accuracy = []
    F1_score = []
    relevant_items = sum(Y_valid)

    number_of_steps = 500
    threshold_array = np.asarray(list(range(number_of_steps + 1))) / number_of_steps
    # print(X_valid)
    # print(Y_valid)

    for threshold in tqdm(threshold_array):  # Computing F1 for different threshold
        # test_loss = 0
        # correct = 0
        # true_positives = 0
        # false_and_true_positives = 0
        # relevant_items = 0

        # print(clf.predict_proba(X_test)[:, 1])
        prediction = engine.round_nd_array(clf.predict_proba(X_valid)[:, 1], threshold)
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

        # if threshold in [0.0, 1.0, 0.5, 0.25, 0.75]:
        #     print("\n" + "_" * 100)
        #     print("threshold =\t" + str(threshold))
        #     print("precision =\t" + str(precision[-1]))
        #     print("recall =\t" + str(recall[-1]))
        #     print("F1_score =\t" + str(F1_score))
        #     print("prediction =\t" + str(prediction))
        #     print("Y_valid =\t" + str(Y_valid))
        #     print("true_positives =\t" + str(true_positives))
        #     print("false_and_true_positives =\t" + str(false_and_true_positives))
        #     print("relevant_items =\t" + str(relevant_items))
        #     print("_" * 100)

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

    plt.savefig("../Archive/F1_score_function of a SVM V_Test-1.0.pdf", dpi=300)
    plt.show()

    finish_learning = time.perf_counter()
    print("Learning was finished in {} seconds time".format(finish_learning - start_learning))
