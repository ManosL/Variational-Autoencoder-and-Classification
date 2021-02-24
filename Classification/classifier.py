import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import Sequential

from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from hyperparameters import Hyperparameters

import math
import input_fns

def classifier(hyperparameters, encoder):
    classifier_model = keras.Sequential()

    index = len(encoder.layers) - 1
    last_layer = encoder.get_layer(index=index)
    #print("index: ", index)
    
    classifier_model.add(layers.Flatten(input_shape=last_layer.output_shape[1:])) # flatten output of encoder

    if hyperparameters.dropout[0] != 0.0:
        classifier_model.add(Dropout(hyperparameters.dropout[0], name='dropoutfl'))

    classifier_model.add(Dense(hyperparameters.neurons, activation = "relu")) # input_shape,


    if hyperparameters.dropout[1] != 0.0:
        classifier_model.add(Dropout(hyperparameters.dropout[1], name='dropoutfc')) # between fc,softmax

    classifier_model.add(Dense(10, activation = "softmax"))

    return classifier_model

def classifier_hyperparameters():
    to_give_neurons = input("Do you want to set neurons in FC layer(y/n) ")

    while to_give_neurons != "y" and to_give_neurons != "n":
        print("")
        print("Invalid answer")
        to_give_neurons = input("Do you want to set neurons in FC layer(y/n) ")


    neurons = 64
    if to_give_neurons == "y":
        print("")
        neurons = int(input("Give number of neurons: "))
        while neurons <= 0:
            print("")
            print("Invalid answer")
            neurons = int(input("Give number of neurons: "))

    print("")
    to_give_dropout = input("Do you want to add a Dropout layer in Flatten layer(y/n) ")

    dropouts = []

    for i in range(2):
        while to_give_dropout != "y" and to_give_dropout != "n":
            print("")
            print("Invalid answer")
            if i == 0:
                to_give_dropout = input("Do you want to add a Dropout layer in Flatten layer(y/n) ")
            else:
                to_give_dropout = input("Do you want to add a Dropout layer in FC layer(y/n) ")

        if to_give_dropout == "n":
            dropouts.append(0.0)
        else:
            dropout_rate = float(input("Give dropout rate of Dropout's layer: "))

            while (dropout_rate <= 0.0 or dropout_rate >= 1.0):
                print("")
                print("Invalid answer(should be between 0.0 and 1.0")
                dropout_rate = float(input("Give dropout rate of Dropout's layer: "))
            dropouts.append(dropout_rate)

        if i == 0:
            print("")
            to_give_dropout = input("Do you want to add a Dropout layer in FC layer(y/n) ")


    print("")
    epochs = input_fns.input_epochs()
    print("")
    batch_size = input_fns.input_batch_size()
    print("")

    return Hyperparameters(0, 0, 0, dropouts,
                        0, 0, epochs, batch_size, neurons)


#index of last conv in encoder
#find all letters of one string inside another string
def last_convlayer(autoencoder):
    index = len(autoencoder.layers) - 1

    # last layer of encoder
    return math.floor(index/2) - 1

    """str = "max_pooling2d"

        for i in range(math.floor(index/2) - 1, -1, -1):
        enc_layer = autoencoder.get_layer(index = i)
        enc_str = enc_layer.name

        if all([c in enc_str for c in str]):
            return i
    return -1"""

# split trained encoder from decoder
def split_model(model, index):
    enc_input = tf.keras.Input(shape=(28,28,1))#comment
    enc_model = enc_input

    #check if weights remain otherwise change input-index=0 and output
    for layer in model.layers[0:index+1]:
        enc_model = layer(enc_model)
    
    enc_model = Model(inputs = enc_input, outputs = enc_model)

    return enc_model


def merge_models(encoder, classifier):
    enc_input = tf.keras.Input(shape=(28,28,1))
    merged_model = enc_input

    for layer in encoder.layers[1:]:
        merged_model = layer(merged_model)

    for layer in classifier.layers:
        merged_model = layer(merged_model)

    merged_model = Model(inputs = enc_input, outputs = merged_model)

    return merged_model


def classifier_train(classifier_model, train_images_code, train_labels, hyperparams):
    classifier_model.compile(loss=losses.CategoricalCrossentropy(),
                                optimizer="RMSprop", metrics=[metrics.CategoricalAccuracy()])


    train_X, test_X, train_ground, test_ground = train_test_split(train_images_code, train_labels,
                                                                    test_size=0.2)


    train_hist = classifier_model.fit(train_X, train_ground, batch_size=hyperparams.batch_size,verbose=1,
                                epochs=hyperparams.epochs, validation_data=(test_X, test_ground))

    return classifier_model, train_hist


def merged_modeltrain(encoder,classifier_model, train_images, train_labels):
    merged_model = merge_models(encoder, classifier_model)
    merged_model.compile(loss=losses.CategoricalCrossentropy(),
                        optimizer="RMSprop", metrics=[metrics.CategoricalAccuracy()])
                        
    #print(merged_model.summary())
    train_X, test_X, train_ground, test_ground = train_test_split(train_images, train_labels,
                                                                    test_size=0.2)
                                                                    
    print("For merged model:")
    epochs = input_fns.input_epochs()
    batch_size = input_fns.input_batch_size()

    train_hist = merged_model.fit(train_X, train_ground, batch_size=batch_size,verbose=1,
                                epochs=epochs, validation_data=(test_X, test_ground))

    return merged_model, train_hist
