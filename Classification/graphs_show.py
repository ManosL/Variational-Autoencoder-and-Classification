import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def loss_epoch_graph(train_loss, val_loss):
    plt.plot(list(range(len(train_loss))), train_loss, label="Train Loss")
    plt.plot(list(range(len(val_loss))), val_loss, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    
    plt.title("Loss change at " + str(len(val_loss)) + " epochs")
    plt.legend()
    plt.show()

    return

def acc_epoch_graph(train_acc, val_acc):
    plt.plot(list(range(len(train_acc))), train_acc, label="Train Accuracy")
    plt.plot(list(range(len(val_acc))), val_acc, label="Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    
    plt.title("Accuracy score change at " + str(len(val_acc)) + " epochs")
    plt.legend()
    plt.show()

    return

def classifier_results(train_vectors, true_labels, pred_labels):
    rows                   = 3
    columns                = 4
    max_indexes_num        = rows * columns

    correct_pred_indexes   = []
    incorrect_pred_indexes = []

    for i in range(len(true_labels)):
        if (len(correct_pred_indexes) >= max_indexes_num) and\
            (len(incorrect_pred_indexes) >= max_indexes_num):
            break
            
        if true_labels[i] == pred_labels[i]:
            if len(correct_pred_indexes) < max_indexes_num:
                correct_pred_indexes.append(i)
        else:
            if len(incorrect_pred_indexes) < max_indexes_num:
                incorrect_pred_indexes.append(i)
        
    fig=plt.figure(figsize=(10, 10))
    rows = 3
    columns = 4
    for i in range(1, columns*rows +1):
        vector_index = correct_pred_indexes[i - 1]

        ax0 = fig.add_subplot(rows, columns, i)
        ax0.imshow(np.array(train_vectors[vector_index]).reshape((28,28)))
        ax0.set_title("Predicted: " + str(pred_labels[vector_index]) + ". Actual Class: " + str(true_labels[vector_index]), fontsize = 10)

    plt.tight_layout(h_pad = 2.0)
    fig.suptitle("Correct Predictions")
    plt.show()


    fig=plt.figure(figsize=(10, 10))
    rows = 3
    columns = 4
    for i in range(1, columns*rows +1):
        vector_index = incorrect_pred_indexes[i - 1]

        ax0 = fig.add_subplot(rows, columns, i)
        ax0.imshow(np.array(train_vectors[vector_index]).reshape((28,28)))
        ax0.set_title("Predicted: " + str(pred_labels[vector_index]) + ". Actual Class: " + str(true_labels[vector_index]), fontsize = 10)

    plt.tight_layout(h_pad = 2.0)
    fig.suptitle("Incorrect Predictions")
    plt.show()

    return 1