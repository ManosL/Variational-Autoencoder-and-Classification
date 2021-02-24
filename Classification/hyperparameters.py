import input_fns

class Hyperparameters:
    def __init__(self, layers, filter_height,filter_width, dropout,
                max_pooling, filters_num, epochs, batch_size, neurons=64):

        self.layers        = layers
        self.filter_height = filter_height
        self.filter_width  = filter_width
        self.dropout       = dropout
        self.max_pooling   = max_pooling
        self.filters_num   = filters_num
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.neurons       = neurons

        return
