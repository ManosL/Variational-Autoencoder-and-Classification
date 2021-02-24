from tensorflow.keras import Sequential
from tensorflow.keras import layers

from hyperparameters import Hyperparameters

# Functions that create the model

def encoder(model, hyperparameters):
    model.add(layers.Conv2D(hyperparameters.filters_num[0],
        (hyperparameters.filter_height, hyperparameters.filter_width),
        activation="relu", padding="same", input_shape=(28,28,1))) # I will check some actvtn funs

    model.add(layers.BatchNormalization())
    
    # If the user gave max pooling
    if hyperparameters.max_pooling[0] != (0,0):
        model.add(layers.MaxPooling2D(hyperparameters.max_pooling[0]))
    
    # If the user gave dropout
    if hyperparameters.dropout[0] != 0.0:
        model.add(layers.Dropout(hyperparameters.dropout[0]))
    
    for i in range(1, hyperparameters.layers):
        model.add(layers.Conv2D(hyperparameters.filters_num[i],
                (hyperparameters.filter_height, hyperparameters.filter_width),
                activation="relu", padding="same")) # I will check some actvtn funs

        model.add(layers.BatchNormalization())

        if hyperparameters.max_pooling[i] != (0,0):
            model.add(layers.MaxPooling2D(hyperparameters.max_pooling[i]))
        
        if hyperparameters.dropout[i] != 0.0:
            model.add(layers.Dropout(hyperparameters.dropout[i]))        
    
    return model

def decoder(model, hyperparameters):
    # Just does the opposite process of encoder's function
    # in ordder to achieve a mirrored architecture
    
    for i in range(hyperparameters.layers):
        model.add(layers.Conv2D(hyperparameters.filters_num[-(i+1)],
                (hyperparameters.filter_height, hyperparameters.filter_width),
                activation="relu", padding="same")) # I will check some actvtn funs

        model.add(layers.BatchNormalization())

        # check if these 2 ifs I do them with the right order
        if hyperparameters.max_pooling[-(i+1)] != (0,0):
            model.add(layers.UpSampling2D(hyperparameters.max_pooling[-(i+1)]))
        
        if hyperparameters.dropout[-(i+1)] != 0.0:
            model.add(layers.Dropout(hyperparameters.dropout[-(i+1)]))

    model.add(layers.Conv2D(1, (hyperparameters.filter_height, hyperparameters.filter_width),
            activation="sigmoid", padding="same"))
    
    return model

def autoencoder(hyperparameters):
    model = Sequential()

    model = encoder(model, hyperparameters)
    model = decoder(model, hyperparameters)

    return model