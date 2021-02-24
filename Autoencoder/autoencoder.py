import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import sys

from autoencoder_model import autoencoder
from hyperparameters import Hyperparameters, get_hyperparameters

from input_fns import input_show_loss_graph, input_save_weights
from input_fns import input_train_again

from graphs_show import loss_epoch_graph, r2_epoch_graph, autoencoder_results

def r2_score(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Reading the dataset file to create tensors ready for input
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

                curr_row.append(np.array(np.array([curr_pixel / 255])))

            curr_vector.append(np.array(curr_row))

        assert (len(curr_vector) == vector_rows)
        
        for i in curr_vector:
            assert(len(i) == vector_cols)

        vectors.append(np.array(curr_vector))

    assert(len(vectors) == dataset_rows)

    return np.array(vectors)

def main():
    argv = sys.argv

    if len(argv) != 3:
        print("Usage: python3 autoencoder.py -d <dataset path>")
        return 1

    dataset_path = ""    
    if argv[1] == "-d":
        dataset_path = argv[2]
    else:
        print("Usage: python3 autoencoder.py -d <dataset path>")
        return 1

    dataset_file = open(dataset_path, "rb")

    train_vectors = readDataset(dataset_file)

    dataset_file.close()

    hyperparams_used   = []
    
    exprmnt_train_loss = []
    exprmnt_val_loss   = []

    exprmnt_train_r2   = []
    exprmnt_val_r2     = []

    to_stop = False

    while (not to_stop):
        hyperparams       = get_hyperparameters()

        autoencoder_model = autoencoder(hyperparams)

        autoencoder_model.summary() #If you want to check model's architecture

        # Checking if the user gave invalid hyperparameters
        if autoencoder_model.layers[-1].output_shape != (None, 28,28,1):
            print("The hyperparameters you gave are wrong because")
            print("the output shape of final layer should be the same")
            print("as the shape of input")
            
            return 1

        autoencoder_model.compile(loss="mean_squared_error", optimizer="RMSprop", metrics=[r2_score])

        train_X, test_X, train_ground, test_ground = train_test_split(train_vectors, train_vectors,
                                                                    test_size=0.2)
        

        train_hist = autoencoder_model.fit(train_X, train_ground, batch_size=hyperparams.batch_size,verbose=1,
                                epochs=hyperparams.epochs, validation_data=(test_X, test_ground))

        train_loss = train_hist.history["loss"]
        val_loss   = train_hist.history["val_loss"]
        train_r2   = train_hist.history["r2_score"]
        val_r2     = train_hist.history["val_r2_score"]

        hyperparams_used.append(hyperparams)

        exprmnt_train_loss.append(train_loss[-1])
        exprmnt_val_loss.append(val_loss[-1])

        exprmnt_train_r2.append(train_r2[-1])
        exprmnt_val_r2.append(val_r2[-1])

        autoencoder_results(train_vectors[5000:], 
                            autoencoder_model.predict(train_vectors[5000:]))

        assert len(train_loss) == hyperparams.epochs

        show_loss_graph = input_show_loss_graph()
        
        if show_loss_graph == True:
            loss_epoch_graph(hyperparams, train_loss, val_loss)
            r2_epoch_graph(hyperparams, train_r2, val_r2)

        print("")

        input_save_weights(autoencoder_model)
        
        print("")

        train_again = input_train_again()

        if train_again == False:
            to_stop = True

    return 0

if __name__ == "__main__":
    main()