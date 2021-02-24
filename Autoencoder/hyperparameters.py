import input_fns

class Hyperparameters:
    def __init__(self, layers, filter_height,filter_width, dropout, 
                max_pooling, filters_num, epochs, batch_size):
        
        self.layers        = layers
        self.filter_height = filter_height
        self.filter_width  = filter_width
        self.dropout       = dropout
        self.max_pooling   = max_pooling
        self.filters_num   = filters_num
        self.epochs        = epochs
        self.batch_size    = batch_size

        return

def get_hyperparameters():
    layers = input_fns.input_layers_num()

    print("")

    filter_height, filter_width = input_fns.input_filter_size()

    print("")

    filters_num = []
    max_pools   = []
    dropouts    = []

    for i in range(layers):
        filters_num.append(input_fns.input_layer_filters_no(i + 1))

        print("")

        max_pools.append(input_fns.input_max_pool_dims(i + 1))

        print("")

        dropouts.append(input_fns.input_dropout_rate(i + 1))
        
    
    print("")

    epochs     = input_fns.input_epochs()

    print("")

    batch_size = input_fns.input_batch_size()

    print("")

    return Hyperparameters(layers, filter_height, filter_width, dropouts, 
                        max_pools, filters_num, epochs, batch_size)