import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def loss_epoch_graph(hyperparams, train_loss, val_loss):
    assert (len(train_loss) == hyperparams.epochs)\
             and (len(val_loss) == hyperparams.epochs)

    plt.plot(list(range(hyperparams.epochs)), train_loss, label="Train Loss")
    plt.plot(list(range(hyperparams.epochs)), val_loss, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    
    plt.title("Loss change at " + str(hyperparams.epochs) + " epochs")
    plt.legend()
    plt.show()

    return

def r2_epoch_graph(hyperparams, train_r2, val_r2):
    assert (len(train_r2) == hyperparams.epochs)\
             and (len(val_r2) == hyperparams.epochs)

    plt.plot(list(range(hyperparams.epochs)), train_r2, label="Train R^2")
    plt.plot(list(range(hyperparams.epochs)), val_r2, label="Validation R^2")

    plt.xlabel("Epochs")
    plt.ylabel("R^2")
    
    plt.title("R^2 score change at " + str(hyperparams.epochs) + " epochs")
    plt.legend()
    plt.show()

    return

def autoencoder_results(train_vectors, pred_vec):
    fig = plt.figure(figsize=(10,10))
    gs0 = gridspec.GridSpec(5, 3) # initial subplots space

    for i in range(12):
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[i], wspace=0.05) # subplot inside grid's space(which is subplots)

        ax0 = fig.add_subplot(gs00[0])
        ax0.imshow(np.array(train_vectors[i]).reshape((28,28)))
        ax0.set_title('Train', fontsize = 10)

        ax1 = fig.add_subplot(gs00[1])
        ax1.imshow(pred_vec[i].reshape((28, 28)))
        ax1.set_title("Predict" , fontsize = 10)


    plt.tight_layout(h_pad = 3.0)
    plt.show()

    return 1
