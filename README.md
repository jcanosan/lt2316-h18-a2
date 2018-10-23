# LT2316 H18 Assignment 2

Git project for implementing assignment 2 in [Asad Sayeed's](https://asayeed.github.io) machine learning class in the University of Gothenburg's Masters
of Language Technology programme.


##Instructions to run the code.
For running train.py, the following arguments are needed:\
python3 train.py --option --maxinstances checkpointdir modelfile categories\
- option must be B (-PB).
- maxinstances is the maximum number of instances of MS COCO for training.\
Warning: If it is set to train more than 400,000 captions, the program raises a MemoryError
when building categorical one-hot vectors that are needed to train the softmax output.
- checkpointdir is the directory where the model checkpoints are saved.
- modelfile is the name of the file in which the model will be saved. This file will be 
inside the folder "models/", but this folder does not have to be specified.
- categories are the (2 or more) categories chosen to train the model.

For running test.py, the following arguments are needed:\
python3 test.py --option --maxinstances modelfile categories caption
- option must be B (-PB).
- maxinstances is the maximum number of instances of MS COCO for testing.\
- modelfile is the name of the file with the trained model. This file will be inside 
the folder "models/", but this folder has not to be specified.
- categories are the (2 or more) categories chosen to test the model. For testing the 
models that I have trained, these must be 'laptop' and 'bird'.
- caption is a string with an incomplete caption to predict the last word and categories.

##Report of the assignment.
The main recurring neural network (RNN) architecture chosen for this project has 
the following structure:

![Alt text](img/LSTMDiagram.jpg?raw=true)

Let's see these layers in more detail:
- Embedding.\
This layer is to encode (therefore, reduce) the dimensionality of the input from 
10000 (the vocabulary size) dimensions to 100.

- Long Short-Term Memory (1) (LSTM).\
The two LSTM layers are the "core" of this model. LSTM is a good Recurrent Neural Network to
learn long-term dependencies because it compares constantly the new inputs to the layer with 
the old outputs and learns from the necessary information of those old outputs.\
I needed to turn return_sequences=True because I have a second LSTM layer in my architecture.
When this argument is True, the output of the LSTM is a 3D array instead of a 2D. This 
dimensionality is required as an input of an LSTM layer.
\[https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47]

- Dropout.\
The dropout layer is useful to prevent overfitting by "randomly setting a fraction 
rate of input units to 0" (from Keras documentation).
To be sincere, there has been no special reason for choosing a 0.1 rate apart from 
the willing to experiment with this layer.

- Long Short-Term Memory (2).\
A second stacked LSTM layer to add depth to the model. Adding additional hidden layers is
"understood to recombine the learned representation from prior layers and create 
new representations at high levels of abstraction". This should make the model better, in theory.\
\[https://machinelearningmastery.com/stacked-long-short-term-memory-networks/]\
Again, I turn True return_sequences. I do so because the input of the following 
following layers of the two branches needs to be 3D.

From here, my model splits into two branches.
- For the last predicted word in caption:
    - Dense.\
    The Dense layer (aka fully connected layer outside Keras) performs a linear 
    matrix vector multiplication in which every input is connected to every output 
    by a weight. It is used to change the dimensions of the vector to 2D.
    - Softmax activation.\
    This layer applies the softmax activation function to the output.\
    This softmax activation function outputs a distribution on the possible last 
    words of the input captions (encoded as one-hot categorical vectors). 
    All the entries of this distribution add up to 1.
- For the category prediction:
    - Long Short-Term Memory (3).\
    Another LSTM, this time to reduce the dimensionality of the model because the output
    is a 2D sigmoid vector. That's why return_sequences=False in this layer.
    - Sigmoid activation.\
    This layer applies the sigmoid activation function to the output.\
    The sigmoid activation function is good for classification since it makes 0 to 1 
    predictions for each of the possible classes.
    This output is a 2D vector with 0 or 1 values for each category of each 
    caption. The indexes of this values match with the indexes of the full list 
    of categories.

As for the compiler, I have used:
- Softmax output: categorical cross-entropy loss function.\
This loss function measures the performance of a model whose output is a probability 
value between 0 and 1, which is what results from the outputs of this model. The more 
the predicted probability diverges from the observed label, the higher is the loss measure.\
\[https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html]
- Sigmoid output: binary cross-entropy loss function.\
This loss function does the same as the categorical one, but only with two possible
labels. In our data, those two labels are 0 and 1 for each category.
- RMSprop.\
RMSprop adjusts automatically que learning rate by updating that rate for each parameter.\
For the preliminary training of the model in 400,000 captions of all the categories of the 
dataset, the learning_rate=0.001, while in the retraining for the specified categories
captions (33,820) the learning_rate=0.002. 
This difference is due to the fact that it's more interesting for our model that 
it learns more from the specific categories than from the general initial train, which
is only done because LSTM needs a lot of data to train a model that performs well.
\[https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/]

When fitting the model:\
- Epochs = 10. I suppose I should have done more (at least 30 or 40, I guess) to train a
more optimized model, but that was not possible due to time constrictions.
- Batch size = 100. There is no specific reason for choosing to use 100 training 
examples on each iteration, I really don't if it is big enough.

##Evaluation:

The model file of the main model is: lstm+dropout-lr0.002.h5

The following are the values for the loss and accuracy metrics of my model:\
Last word (y) loss (categorical cross-entropy) = 1.943227\
Last word (y) accuracy = 0.627255\
Multiclass category loss (binary cross-entropy) = 0.351842\
Multiclass category accuracy = 0.955727

On the one hand, the word predictor seems to have problems, looking at the results. 
The accuracy is quite low while the loss seems high.
On the other hand, the multiclass classificator shows some really good results with a 
low loss and a high accuracy.

Looking at these results, we should expect a good amount of mistaken last words and 
a good number of correct categories when predicting from a caption. But in fact, it seems
to happen the other way in the tested example captions (two for the 'birds' category, and 
two for the 'laptop' category).

The following examples are an excerpt that represent what I have seen in the multiple 
text predictions that I have done.

Example 1: "an up close shot of a seagull flying in front of the"\
Input: an up close shot of a seagull flying in front of the\
Last word: ocean. (the one in the dataset was "beach")\
Top 5 categories with scores:\
	person: 0.269629\
	cup: 0.268977\
	horse: 0.268976\
	potted plant: 0.268965\
	laptop: 0.268950
	    
Example 2: "an unusual bird walks the plains with wild"\
Input: an unusual bird walks the plains with wild\
Last word: animals. (in the dataset: "buffalo")\
Top 5 categories with scores:\
	horse: 0.269029\
	hair drier: 0.269020\
	person: 0.269000\
	laptop: 0.268993\
	refrigerator: 0.268955

Example 3: "a cat laying in front of a computer that is turned"\
Input: a cat laying in front of a computer that is turned\
Last word: on. (this is the original word in the dataset)\
Top 5 categories with scores:\
	laptop: 0.500000\
	keyboard: 0.290477\
	book: 0.270161\
	chair: 0.269016\
	cup: 0.268971

Example 4: "a living room has a couch, table, and working"\
Input: a living room has a couch, table, and working\
Last word: television. (this is the original word in the dataset)\
Top 5 categories with scores:\
	chair: 0.731042\
	tv: 0.500000\
	laptop: 0.500000\
	couch: 0.500000\
	book: 0.500000

These examples are extracted from the COCO website. Therefore, they are part of the 
training data. Testing on the train data is not a good practice because the training 
data will be always have a high probability of be assigned correctly during a test and
the scores will be very high when evaluating.

Even though, in this tests can be seen that the predictions tend to have a good amount
of errors.\
The words, even if some are not guessed as they were in the original dataset (which can
explain the accuracy and loss metrics), it tends to predict words that are correct in a
syntactic way. Therefore, it seems that the word prediction behaves pretty well.\
Looking at the top categories, they are totally messed for the first two 
captions: they are incorrect and have very low scores, which reveals that the model has not
managed to predict the categories for those captions, not even the main one ('bird').
For the last two examples the situation changes. In the first one it guesses correctly
"laptop", but that's the only one (in the original image there is also a cat).
Even if the keyboard is guessed as the second category, the score is very low.
The prediction for the last one has been totally different. Here it has guessed 
a good amount of categories with high scores and all of them are correct according to the
original image.\
In sum, while the last word prediction seems fairly good, the classifier seems quite more
inconsistent in its results even for the training data.

This inconsistency may be due to the fact that the training data is not too large. 
Even if the preliminary training in captions of the whole dataset is done in 400,000 
captions, the retraining in the specific categories is done in 33,820 captions.


##Architectural variant of my network
As an architectural variant for my network, I have deleted of the second LSTM layer
of my RNN. Doing so, I reduce the depth of my model, which (in theory) should make my
model worse.

This variant is in the model file: lstm+dropout-lr0.002-nolstm2.h5

Last word (y) loss (categorical cross-entropy) = 1.959088\
Last word (y) accuracy = 0.625352\
Multiclass category loss (binary cross-entropy) = 0.346684\
Multiclass category accuracy = 0.962386

Input: an up close shot of a seagull flying in front of the\
Last word: ocean.\
Top 5 categories with scores:\
	bird: 0.500000\
	toilet: 0.269306\
	fire hydrant: 0.269231\
	person: 0.269047\
	tv: 0.269046

Input: an unusual bird walks the plains with wild\
Last word: animals\
Top 5 categories with scores:\
	bird: 0.500000\
	person: 0.327339\
	laptop: 0.269198\
	fire hydrant: 0.268988\
	clock: 0.268978

Input: a cat laying in front of a computer that is turned\
Last word: on.\
Top 5 categories with scores:\
	laptop: 0.730191\
	cat: 0.362283\
	tv: 0.330522\
	mouse: 0.272202\
	carrot: 0.270123

Input: a living room has a couch, table, and working\
Last word: computer.\
Top 5 categories with scores:\
	laptop: 0.731044\
	tv: 0.591155\
	couch: 0.500000\
	chair: 0.500000\
	book: 0.500000
	
The loss and accuracy scores show almost no difference and the last word prediction
behaves in almost the same way. It only changes in the last example, where it predicts the
word that is not in the original caption, but even then is a consistent word for this sentence.\
The classifier clearly has improved. Now it manages to predict at least the trained category 
('bird') on the first two ones and to place in the second position 'cat' in the third example,
even if the score is not really high. In the last one it has switched some positions, placing
the 'laptop' first.

In conclusion, it seems that the model performs better with less depth, which is something
that I was not expecting when I designed my LSTM neural network.


##One learning_rate variant
Firstly, I know that I should have test two variants, but I have not done so due to time 
constrictions.

This variant is in the model file: lstm+dropout-lr0.01.h5

I have decided to try to increase the learning_rate of the second train in the specified 
categories from 0.002 to 0.01, which seems a huge difference. I expect then that my model 
will learn more then from the specific categories, which seems more interesting for this model.
Therefore, maybe it clan classify better the captions of birds that were failing in the main
model. Let's see it this has improved.

Last word (y) loss (categorical cross-entropy) = 2.026975\
Last word (y) accuracy = 0.620333\
Multiclass category loss (binary cross-entropy) = 0.351549\
Multiclass category accuracy = 0.954307

The measures from the evaluation do not look bad. The loss of the last word is slightly high
and the accuracy slightly low, but does not seem an important difference.

Input: an up close shot of a seagull flying in front of the\
Last word: sky\
Top 5 categories with scores:\
	bird: 0.500000\
	person: 0.269331\
	elephant: 0.268944\
	traffic light: 0.268944\
	motorcycle: 0.268944

Input: an unusual bird walks the plains with wild\
Last word: flowers.\
Top 5 categories with scores:\
	bird: 0.500000\
	person: 0.268986\
	motorcycle: 0.268951\
	traffic light: 0.268948\
	elephant: 0.268947

Input: a cat laying in front of a computer that is turned\
Last word: on.\
Top 5 categories with scores:\
	laptop: 0.500000\
	bird: 0.500000\
	book: 0.401676\
	chair: 0.305901\
	potted plant: 0.268949
	
Input: a living room has a couch, table, and working\
Last word: computer.\
Top 5 categories with scores:\
	chair: 0.730788\
	tv: 0.500000\
	laptop: 0.500000\
	couch: 0.500000\
	book: 0.500000
	
For the last word prediction it does not seem that it has change really much the behaviour, even if
some words are different.\
For the bird images it seems that the learning rate change has been good, since now it is able
to predict the main category with a good score. Also, the categories for the fourth caption 
have not change.
However, the third example has included the category 'bird', which is clearly wrong for this image.
I would say that maybe the huge difference in the learning rate for the specific images in relation 
to the first general training has made the model to give higher scores to the two specified 
categories, but looking at the rest of examples that does not seem to be the case.


##What I have learned with this assignment
As it has been my first contact with neural networks and Keras, it has been really hard but in
the end I have learned a lot about how to design, run and test them. 
Among other things, this is what I think that is the main knowledge that I have acquired:

- Preprocess the data for the tasks of word prediction and multiclass classification.
- Design a RNN (LSTM) at a basic level.
- Choose the needed layers to deal with the shapes and dimensionalities.
- Investigate the basic differences between the different optimizers and loss functions and choose
which seems to fit the best for my task.
- Hyperparameters of the Neural Networks.
- Sigmoid and softmax outputs.
- Evaluate, predict and decode the predictions to turn them into readable words.