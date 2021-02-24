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
    show_loss_graph = input("Do you want to show the loss and accuracy graphs?(y/n) ")

    while (show_loss_graph != "y") and (show_loss_graph !="n"):
        print("")
        print("Invalid answer")
        show_loss_graph = input("Do you want to show the loss graphs?(y/n) ")
    
    if show_loss_graph == "y":
        return True
    else:
        return False

def input_test_set_eval():
    test = input("Do you want to evaluate the model on test set?(y/n) ")

    while (test != "y") and (test != "n"):
        print("")
        print("Invalid answer")
        test = input("Do you want to evaluate the model on test set?(y/n) ")
    
    if test == "n":
        return None

    print()
    print("Do you want to evaluate test set on pretrained-encoder")
    model_choice = int(input("and trained classifier(choice 1) or in the merged model(choice 2)? "))

    while (model_choice != 1) and (model_choice != 2):
        print("Invalid answer")
        print("Do you want to evaluate test set on pretrained-encoder")
        model_choice = int(input("and trained classifier(choice 1) or in the merged model(choice 2)? "))

    return model_choice

def input_save_as_clusters():
    to_save = input("Do you want to write train set in a file as clusters?(y/n) ")

    while (to_save != "y") and (to_save != "n"):
        print("")
        print("Invalid answer")
        to_save = input("Do you want to write train set in a file as clusters?(y/n) ")

    if to_save == "y":
        path = input("Give the path to save the training set as clusters: ")
        return path
    else:
        return None

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
