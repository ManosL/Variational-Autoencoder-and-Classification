# Variational Autoencoder and Classification

# Authors

The project was written by the following:

  1. [Emmanouil Lykos](https://github.com/ManosL)
  2. [Apostolos-Nikolaos Mponotis](https://github.com/AkisBon)

**Note**: The results and their evaluation are on Experiments.pdf

# Autoencoder

### Manual

In order to run the autoencoder you should have as working
directory the Autoencoder directory and run on the terminal
the following command:

`python3 autoencoder.py -d <dataset file>`

where:

  1. <dataset file> is the path to the dataset that we will 
     evaluate the user's autoencoder model.

Afterwards, the user should provide the hyperparameters that 
the program asks. These are the following:

  1. The number of encoder's convolutional layers.

  2. The size of convolutional layer's kernel. The recommended
  height and width value is 3 because we have 28x28 pictures.

  3. For each layer asks the number of its filters and if they
  will have afterwards a max pooling or/and a dropout layer.

  4. The number of training epochs.

  5. The batch size.

After training the user can show the loss and R2 score change 
through epochs, save the model into an h5 file and run another
experiment with different hyperparameters.

### Description

After the user gave the necessary hyperparameters, the corresponding
autoencoder is built. Note that, its architecture form is mirrored, which
is that after the encoder's layers we have their mirrored version that
represent the decoder. Then, the training split is splitted into a new
training set and a validation set and the network is trained. After, the
user can do what is mentioned in the Manual.

# Classifier

### Manual

In order to run the classifier you should have as working
directory the Classification directory and run on the 
terminal the following command:

  `python  classification.py  -d  <training  set>  
  -dl  <training  labels> -t <test set> 
  -tl <test labels> -model <autoencoder h5>`

where:

  1. <training set> is the path to the training set.

  2. <training labels> is the path to the file that 
  contains the labels of <training set>.

  3. <test set> is the path to the test set.

  4. <test labels> is the path to the file that 
  contains the labels of <test set>.

  5. <autoencoder h5> is the path to the h5 file that
  contains the autoencoder model with its corresponding
  pretrained weights.

After the files are read the user should provide if he wants
to give how many neurons the FC layers will have. If he does
not want to provide that the layer will have 64 neurons. Then
he gives the number of training epochs and the batch size. After
the classifier was trained user should give the training epochs and
the batch size in order to train the merged model. Then the user has
the ability to evaluate the classifier on test set and run the 
experiments with different hyperparameters.

### Description

The classifier firstly reads the given files from the command line
and afterwards we create a new model from the encoder's layers we
took from autoencoder and we predict the train vectors(we will tell
them encoded training vectors). Afterwards, the user provides the 
classifier's hyperparameters and the classifier is trained giving
him the encoded training vectors. Then we merge the encoder model
and classifier model and we train it with the new hyperparameters
that the user gave.
        
        
Details:

 classifier.py:

  last_convlayer(autoencoder):

  For every convolution and batch_normalization layers of the encoder,
  we have a pair of convolution and batch_normalization layer for the 
  decoder.

  For every maxpooling layer of the encoder,an up_sampling layer also 
  exists.So there is a symetry.

  Actually,decoder's layers are equal to 2*(number of encoder's layer)+1.
  +1 because decoder has one more last conv2d layer.
  So,half layers - 1 of autoencoder's layers give us the index of the 
  last encoder's layer(code).


  classifier:
  It creates the classifier model asking the user to add dropout
  between Flatten and FC layer(not useful) or FC layer and Dense(softmax) layer.


  classifier_hyperparameters:
  Asks user to set hypermarameters.


  split_model:
  It splits autoencoder returning the encoder part.

  We define an input layer for the new model(shape=(28,28,1)) and 
  create the model sequentially by providing each layer with its 
  previous layer(its input).

  We define the model(Model(inputs,outputs)).


merge_models:

  We define an input layer for the classifier and encoder merged models.

  We create the model sequentially by providing each layer with its
  previous layer(its input).

  We define the model(Model(inputs,outputs)).


classifier_train:

  Compiles the model with proper metrics.

  Splits the data to a test set and a validation test.

  Trains the model.


merge_modeltrain:

  Calls merge_models.

  Compile merged model using classifier's metrics.
  splits the data to a test set and a validation test.

  Trains the test.


 Classification.py:

  Reads training_images set,training_labels,test sets and loads autoencoder model.

  Creates classifier_model and merged_model.

  Trains both models showing the results of each epoch and batch.

  Shows loss and accuracy graphs.

  Evaluetes and predicts the merged model,counting correct and false predictions.

  Uses classifier_results to show the image that classifier predicts labeling it as
  correct or incorrect prediction and also shows the initial image.

  Uses classification_report(sklearn) to count score as well as macro and
  weighted average.
