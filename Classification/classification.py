import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import classifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import sys

from hyperparameters import Hyperparameters
from input_fns import input_show_loss_graph, input_train_again, input_test_set_eval, input_save_as_clusters
from graphs_show import loss_epoch_graph, acc_epoch_graph, classifier_results

import os

def r2_score(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def readDataset(file_obj):
    assert (not file_obj.closed)

    vectors = []            # It will be a list of lists

    _            = file_obj.read(4)
    dataset_rows = int.from_bytes(file_obj.read(4), byteorder="big")
    vector_rows  = int.from_bytes(file_obj.read(4), byteorder="big")
    vector_cols  = int.from_bytes(file_obj.read(4), byteorder="big")

    for _ in range(dataset_rows):
        curr_vector = []

        for _ in range(vector_rows):
            curr_row = []

            for _ in range(vector_cols):

                curr_pixel  = int.from_bytes(file_obj.read(1), byteorder="big", signed=False)

                assert ((curr_pixel >= 0) and (curr_pixel <= 255))

                curr_row.append(np.array([curr_pixel / 255]))

            curr_vector.append(np.array(curr_row))

        assert (len(curr_vector) == vector_rows)

        for i in curr_vector:
            assert(len(i) == vector_cols)

        vectors.append(np.array(curr_vector))

    assert(len(vectors) == dataset_rows)

    return np.array(vectors)

def readLabels(file_obj):
    assert (not file_obj.closed)

    oneHotLabels = []            # It will be a list of lists

    _            = file_obj.read(4)
    labels_num   = int.from_bytes(file_obj.read(4), byteorder="big")
    total_labels = 10

    for _ in range(labels_num):
        curr_label = int.from_bytes(file_obj.read(1), byteorder="big")
        oneHotLabels.append(np.array([0] * (curr_label) + [1] + [0] * (total_labels - curr_label - 1)))

    return np.array(oneHotLabels)

def getPredictedLabels(netPreds):
    pred_labels = []

    for pred in netPreds:
        pred_labels.append(np.argmax(pred))

    return np.array(pred_labels)

def getCorrectFalseNum(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    correct = 0
    false   = 0

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
        else:
            false += 1

    assert (correct + false) == len(y_true)

    return correct, false

def main():
    argv = sys.argv

    if len(argv) != 11:
        print("Usage: python3  classification.py  –d  <training  set>  –dl  <training  labels>")
        print("-t <testset> -tl <test labels> -model <autoencoder h5>")

        return 1

    training_set_path        = None
    training_labels_path     = None
    test_set_path            = None
    test_labels_path         = None
    autoencoder_weights_path = None

    for i in range(1, len(argv), 2):
        arg = argv[i]

        if arg == "-d":
            if training_set_path != None:
                print("Cannot give same argument twice")
                print("Usage: python3  classification.py  –d  <training  set>  –dl  <training  labels>")
                print("-t <testset> -tl <test labels> -model <autoencoder h5>")

                return 1

            training_set_path = argv[i+1]
        elif arg == "-dl":
            if training_labels_path != None:
                print("Cannot give same argument twice")
                print("Usage: python3  classification.py  –d  <training  set>  –dl  <training  labels>")
                print("-t <testset> -tl <test labels> -model <autoencoder h5>")

                return 1

            training_labels_path = argv[i+1]
        elif arg == "-t":
            if test_set_path != None:
                print("Cannot give same argument twice")
                print("Usage: python3  classification.py  –d  <training  set>  –dl  <training  labels>")
                print("-t <testset> -tl <test labels> -model <autoencoder h5>")

                return 1

            test_set_path = argv[i+1]
        elif arg == "-tl":
            if test_labels_path != None:
                print("Cannot give same argument twice")
                print("Usage: python3  classification.py  –d  <training  set>  –dl  <training  labels>")
                print("-t <testset> -tl <test labels> -model <autoencoder h5>")

                return 1

            test_labels_path = argv[i+1]
        elif arg == "-model":
            if autoencoder_weights_path != None:
                print("Cannot give same argument twice")
                print("Usage: python3  classification.py  –d  <training  set>  –dl  <training  labels>")
                print("-t <testset> -tl <test labels> -model <autoencoder h5>")

                return 1

            autoencoder_weights_path = argv[i+1]
        else:
            print("Invalid argument")
            print("Usage: python3  classification.py  –d  <training  set>  –dl  <training  labels>")
            print("-t <testset> -tl <test labels> -model <autoencoder h5>")

            return 1

    # Reading the files
    training_set_file    = open(training_set_path, "rb")

    train_vectors        = readDataset(training_set_file)

    training_set_file.close()

    training_labels_file = open(training_labels_path, "rb")

    train_labels         = readLabels(training_labels_file)

    training_labels_file.close()

    test_set_file        = open(test_set_path, "rb")

    test_vectors         = readDataset(test_set_file)

    test_set_file.close()

    test_labels_file     = open(test_labels_path, "rb")

    test_labels          = readLabels(test_labels_file)

    test_labels_file.close()

    to_stop = False
    print("finished reading")

    while (not to_stop):
        dependencies = {
            'r2_score': r2_score
        }

        autoencoder_loaded_model = load_model(autoencoder_weights_path, custom_objects=dependencies)
        autoencoder_loaded_model.summary()
        index = classifier.last_convlayer(autoencoder_loaded_model)

        encoder_model = classifier.split_model(autoencoder_loaded_model, index)
        #encoder_model.summary()

        #code-input for classifier
        train_images_code = encoder_model.predict(train_vectors)

        hyperparams = classifier.classifier_hyperparameters()
        if not os.path.isfile("../Autoencoder/Autoencoder.txt"):
            fp = open("../Autoencoder/Autoencoder.txt","w+")
        else:
            fp = open("../Autoencoder/Autoencoder.txt","a")

        fp.write("\nclassifier:\nepochs: %d, batch: %d, neurons: %d\n" % (hyperparams.epochs, hyperparams.batch_size, hyperparams.neurons))
        fp.write("Flatten layer-dropout: %f\n" % (hyperparams.dropout[0]))
        fp.write("FC layer-dropout: %f\n" % (hyperparams.dropout[1]))
        fp.close()

        classifier_model = classifier.classifier(hyperparams, encoder_model)
        print("")
        #classifier_model.summary()


        #in case predict does not work use comment
        """outputs = []
        for layer in model.layers:
            keras_function = K.function([model.input], [layer.output])
            outputs.append(keras_function([training_data, 1]))
        train_images_code.append[outputs[-1]]
        """

        classifier_model, train_hist = classifier.classifier_train(classifier_model, train_images_code, train_labels, hyperparams)

        merged_model, train_hist = classifier.merged_modeltrain(encoder_model, classifier_model,
                                                                        train_vectors, train_labels)
        #merged_model.summary()

        train_loss = train_hist.history["loss"]
        train_acc  = train_hist.history["categorical_accuracy"]

        val_loss   = train_hist.history["val_loss"]
        val_acc    = train_hist.history["val_categorical_accuracy"]

        show_loss_graph = input_show_loss_graph()

        if show_loss_graph == True:
            loss_epoch_graph(train_loss, val_loss)
            acc_epoch_graph(train_acc, val_acc)

        print("")

        test_set_eval = input_test_set_eval()

        if test_set_eval != None:
            test_vectors_code   = encoder_model.predict(test_vectors)

            testing_vectors     = test_vectors
            testing_model       = merged_model

            if test_set_eval == 1:
                testing_vectors = test_vectors_code
                testing_model   = classifier_model

            #testing_model.summary()
            print()

            test_loss, test_acc = testing_model.evaluate(testing_vectors, test_labels, verbose=0)
            test_preds          = testing_model.predict(testing_vectors)
            test_preds          = getPredictedLabels(test_preds)

            test_true           = getPredictedLabels(test_labels)

            correct_preds, false_preds = getCorrectFalseNum(test_true, test_preds)

            classes             = ["Class " + str(i) for i in range(10)]

            print("Test Loss:", test_loss)
            print("Test Accuracy:", test_acc)
            print()

            print("Correctly predicted", correct_preds, "pictures.")
            print("Falsely predicted", false_preds, "pictures.")
    
            print(classification_report(test_true, test_preds,target_names = classes))
           
            classifier_results(test_vectors, test_true, test_preds)

            print()

        # This input works for us in the 3rd assignment to get the
        # classes as clusters file
        save_as_clusters = input_save_as_clusters()

        if save_as_clusters != None:
            file_obj = open(save_as_clusters, "w")

            preds    = list(getPredictedLabels(merged_model.predict(train_vectors[0:3000])))

            clusts   = {}
            for i in range(10):
                clusts[str(i + 1)] = []

            for i, pr in zip(range(len(preds)), preds):
                clusts[str(pr + 1)].append(i)
            
            for i in range(1, 11):
                file_obj.write("CLUSTER-" + str(i) + " { size: " + str(len(clusts[str(i)])) + ", ")
            
                for j in range(len(clusts[str(i)]) - 1):
                    file_obj.write(str(clusts[str(i)][j]) + ", ")
                
                file_obj.write(str(clusts[str(i)][len(clusts[str(i)]) - 1]) + "}\n")
            
            file_obj.close()
            
        train_again = input_train_again()

        if train_again == False:
            to_stop = True

    return 0

if __name__ == "__main__":
    main()
