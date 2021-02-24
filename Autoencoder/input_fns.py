# General input functions

def input_layers_num():
    layers = int(input("Give the number of encoder's and decoder's convolutional layers: "))

    while layers <= 0:
        print("")
        print("You should give a positive value")
        layers = int(input("Give the number of encoder's amd decoder's convolutional layers: "))
    
    return layers

def input_filter_size():
    filter_height   = int(input("Give filter's height: "))

    while filter_height <= 0:
        print("")
        print("You should give a positive value")
        filter_height   = int(input("Give filter's height: "))

    print("")

    filter_width    = int(input("Give filter's width: "))

    while filter_width <= 0:
        print("")
        print("You should give a positive value")
        filter_width   = int(input("Give filter's width: "))

    return filter_height, filter_width

def input_layer_filters_no(layer_num):
    filters_no = int(input("Give number of filters of layer " + str(layer_num) + ": "))

    while filters_no <= 0:
        print("")
        print("You should give a positive value")
        filters_no = int(input("Give number of filters of layer " + str(layer_num) + ": "))
    
    return filters_no

def input_max_pool_dims(layer_num):
    to_give_max_pool = input("Do you want to add a Max Pooling 2D layer in current Conv layer(y/n) ")

    while to_give_max_pool != "y" and to_give_max_pool != "n":
        print("")
        print("Invalid answer")
        to_give_max_pool = input("Do you want to add a Max Pooling 2D layer in current Conv layer(y/n) ")
    
    if to_give_max_pool == "n":
        return (0, 0)
    
    pool_height   = int(input("Give max pool's layer height: "))

    while pool_height <= 0:
        print("")
        print("You should give a positive value")
        pool_height   = int(input("Give max pool's height: "))

    print("")

    pool_width    = int(input("Give max pool's width: "))

    while pool_width <= 0:
        print("")
        print("You should give a positive value")
        pool_width   = int(input("Give max pool's width: "))

    return (pool_height, pool_width)

def input_dropout_rate(layer_num):
    to_give_dropout = input("Do you want to add a Dropout layer in current Conv layer(y/n) ")

    while to_give_dropout != "y" and to_give_dropout != "n":
        print("")
        print("Invalid answer")
        to_give_dropout = input("Do you want to add a Dropout layer in current Conv layer(y/n) ")
    
    if to_give_dropout == "n":
        return 0.0  

    dropout_rate = float(input("Give dropout rate of Dropout's layer: "))

    while dropout_rate <= 0.0 or dropout_rate >= 1.0:
        print("")
        print("Invalid answer(should be between 0.0 and 1.0")
        dropout_rate = float(input("Give dropout rate of Dropout's layer: "))
    
    return dropout_rate

def input_epochs():
    epochs = int(input("Give number of training epochs: "))

    while epochs <= 0:
        print("")
        print("You should give a positive value")
        epochs          = int(input("Give number of training epochs: "))
    
    return epochs

def input_batch_size():
    batch_size      = int(input("Give the batch size: "))
    
    while batch_size <= 0:
        print("")
        print("You should give a positive value")
        batch_size      = int(input("Give the batch size: "))
    
    return batch_size

def input_show_loss_graph():
    show_loss_graph = input("Do you want to show the loss and R^2 graphs?(y/n) ")

    while (show_loss_graph != "y") and (show_loss_graph !="n"):
        print("")
        print("Invalid answer")
        show_loss_graph = input("Do you want to show the loss graphs?(y/n) ")
    
    if show_loss_graph == "y":
        return True
    else:
        return False

def input_save_weights(autoencoder_model):
    sv_weights = input("Do you want to save model's weights?(y/n) ")

    while (sv_weights != "y") and (sv_weights !="n"):
        print("")
        print("Invalid answer")
        sv_weights = input("Do you want to save model's weights?(y/n) ")
    
    if sv_weights == "y":
        print("")
        save_path = input("Give the path of the file you want to save the weights: ")
        autoencoder_model.save(save_path)
    
    return

def input_train_again():
    train_again = input("Do you want to train again the model with different hyperparameters?(y/n) ")

    while (train_again != "y") and (train_again != "n"):
        print("")
        print("Invalid answer")
        train_again = input("Do you want to train again the model with different hyperparameters?(y/n) ")
    
    if train_again == "y":
        return True
    else:
        return False