# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from argparse import ArgumentParser
import mycoco
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import Model
from keras.layers import Input, Activation, LSTM, Embedding, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import pickle
import os
import numpy as np


# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('train')
    print("Option A not implemented!")


# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB(categories, modelfile, maxinstances):
    mycoco.setmode('train')

    # Extract a list with all the categories.
    all_cats_ids = mycoco.annotcoco.loadCats(mycoco.annotcoco.getCatIds())
    all_categories = [cat['name'] for cat in all_cats_ids]

    # Collect the necessary captions and categories.
    print("Collecting captions and categories...")
    # If it is set to train with more than 400000 captions, it raises a
    # MemoryError when building the categorical one-hot vectors.
    all_caps, all_cats = collect_captions_cats(all_categories, maxinstances)
    caps, cats = collect_captions_cats(categories, maxinstances)
    print("Done!")

    # If there is a tokenizer index saved into a file, load it.
    if os.path.isfile('tokenizer/tokenizer.pickle'):
        print("There is already a tokenizer index of the captions in a file"
              " (tokenizer.pickle). Loading...")
        with open('tokenizer/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Done!")
    # Else: prepare the tokenizer.
    else:
        print("Building the tokenizer index for captions and saving into a file"
              " (tokenizer.pickle)...")
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(all_caps)
        with open('tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done!")

    # Prepare the data
    print("Preparing the captions and categories for modeling...")
    all_caps_X_pad, all_caps_y_categ, all_cats_vec = \
        prepare_data(tokenizer, all_categories, all_caps, all_cats)
    caps_X_pad, caps_y_categ, cats_vec = \
        prepare_data(tokenizer, all_categories, caps, cats)
    print("Data prepared.")

    # Train a language model on the the entire COCO caption set
    X_input_layer = Input(shape=(caps_X_pad.shape[1],))
    emb_layer = Embedding(10000, 100,
                          input_length=caps_X_pad.shape[1])(X_input_layer)
    lstm1 = LSTM(100, return_sequences=True)(emb_layer)
    dropout_layer = Dropout(0.1)(lstm1)
    lstm2 = LSTM(100, return_sequences=True)(dropout_layer)
    dense_layer = Dense(caps_y_categ.shape[2])(lstm2)
    y_softmax_layer = Activation('softmax')(dense_layer)
    lstm3 = LSTM(cats_vec.shape[1])(lstm2)
    cats_sigmoid_layer = Activation('sigmoid')(lstm3)

    model = Model(inputs=[X_input_layer], outputs=[y_softmax_layer,
                                                   cats_sigmoid_layer])
    model.summary()
    optimizer = RMSprop()   # default lr=0.001
    model.compile(loss={'activation_1': 'categorical_crossentropy',
                        'activation_2': 'binary_crossentropy'},
                  optimizer=optimizer, metrics=["accuracy"])

    # Checkpoint
    all_filepath_loss_y = "/scratch/guscanojo/all-y-acc-{epoch:02d}-" \
                          "{activation_1_acc:.2f}.hdf5"
    all_y_loss_checkpoint = ModelCheckpoint(all_filepath_loss_y,
                                            monitor='activation_1_loss',
                                            verbose=1, save_best_only=True,
                                            mode='min')
    all_filepath_loss_cats = "/scratch/guscanojo/all-cats-acc-{epoch:02d}-" \
                             "{activation_2_acc:.2f}.hdf5"
    all_cats_loss_checkpoint = ModelCheckpoint(all_filepath_loss_cats,
                                               monitor='activation_2_loss',
                                               verbose=1, save_best_only=True,
                                               mode='min')
    all_filepath_acc_y = "/scratch/guscanojo/all-y-acc-{epoch:02d}-" \
                         "{activation_1_acc:.2f}.hdf5"
    all_y_acc_checkpoint = ModelCheckpoint(all_filepath_acc_y,
                                           monitor='activation_1_acc',
                                           verbose=1, save_best_only=True,
                                           mode='max')
    all_filepath_acc_cats = "/scratch/guscanojo/all-cats-acc-{epoch:02d}-" \
                            "{activation_2_acc:.2f}.hdf5"
    all_cats_acc_checkpoint = ModelCheckpoint(all_filepath_acc_cats,
                                              monitor='activation_2_acc',
                                              verbose=1, save_best_only=True,
                                              mode='max')
    all_callbacks_list = [all_y_loss_checkpoint, all_cats_loss_checkpoint,
                          all_y_acc_checkpoint, all_cats_acc_checkpoint]

    # Train
    model.fit([all_caps_X_pad], [all_caps_y_categ, all_cats_vec], epochs=10,
              batch_size=100, callbacks=all_callbacks_list)
    
    # Retrain the language model for the specified categories
    # Increase learning rate so it learns more from the specific categories
    optimizer2 = RMSprop(lr=0.002)
    model.compile(loss={'activation_1': 'categorical_crossentropy',
                        'activation_2': 'binary_crossentropy'},
                  optimizer=optimizer2, metrics=["accuracy"])

    # Checkpoint
    filepath_loss_y = "/scratch/guscanojo/specific-y-loss-{epoch:02d}-" \
                      "{activation_1_loss:.2f}.hdf5"
    y_loss_checkpoint = ModelCheckpoint(filepath_loss_y,
                                        monitor='activation_1_loss', verbose=1,
                                        save_best_only=True, mode='min')
    filepath_loss_cats = "/scratch/guscanojo/specific-cats-loss-{epoch:02d}-" \
                         "{activation_2_loss:.2f}.hdf5"
    cats_loss_checkpoint = ModelCheckpoint(filepath_loss_cats,
                                           monitor='activation_2_loss',
                                           verbose=1, save_best_only=True,
                                           mode='min')
    filepath_acc_y = "/scratch/guscanojo/specific-y-acc-{epoch:02d}-" \
                     "{activation_1_acc:.2f}.hdf5"
    y_acc_checkpoint = ModelCheckpoint(filepath_acc_y,
                                       monitor='activation_1_acc', verbose=1,
                                       save_best_only=True, mode='max')
    filepath_acc_cats = "/scratch/guscanojo/specific-cats-acc-{epoch:02d}-" \
                        "{activation_2_acc:.2f}.hdf5"
    cats_acc_checkpoint = ModelCheckpoint(filepath_acc_cats,
                                          monitor='activation_2_acc', verbose=1,
                                          save_best_only=True, mode='max')
    callbacks_list = [y_loss_checkpoint, cats_loss_checkpoint,
                      y_acc_checkpoint, cats_acc_checkpoint]

    # Train
    model.fit([caps_X_pad], [caps_y_categ, cats_vec], epochs=10,
              batch_size=100, callbacks=callbacks_list)

    # Save the model into a file
    model.save("models/" + modelfile)
    print("Model saved in the path:", "models/" + modelfile)


def collect_captions_cats(categories, maxinstances):
    """
    Calls mycoco.py to collect the images ids and uses them to collect the
    captions and categories.
    :param categories: the list of selected categories.
    :param maxinstances: int of the maximum number of captions.
    :return: Two lists: captions and categories.
    """
    ids = mycoco.query(categories, exclusive=False)
    caps, cats = mycoco.list_captions_cats(ids, categories)
    return caps[:maxinstances], cats[:maxinstances]


def prepare_data(tokenizer, all_categories, caps, cats):
    """
    Prepares the tokenizer with all the data.
    Converts the str data into integer sequences and split into the input and
    the two outputs of our model.
    Pads the captions (X) and categories list.
    Converts the three lists into categorical one-hot vectors.

    :param tokenizer: tokenizer index.
    :param all_categories: list with all the categories of the COCO dataset.
    :param caps: list of captions.
    :param cats: list categories.
    :return: the input and the two outputs of the LSTM model:
        caps_X_pad: the padded sequences of all the captions without the last
        word.
        caps_y_categ: categorical one-hot vectors for the softmax output.
        cats_vec: categories 0|1 vectors for the sigmoid output.
    """

    # Call the function to build and split the sequences
    print("\tBuilding sequences and splitting input and outputs...")
    # Sequences
    caps_seqs = tokenizer.texts_to_sequences(caps)
    print("\t\tSequences built.")
    # Split in X and y
    caps_X_seqs = [x[0:-1] for x in caps_seqs]
    caps_y_seqs = [x[1:] for x in caps_seqs]
    print("\t\tSplits done.")
    print("\tSequences done!")

    # Pad the sequences.
    caps_X_pad = pad_sequences(caps_X_seqs, maxlen=15)
    caps_y_pad = pad_sequences(caps_y_seqs, maxlen=15)
    print("\tPad done!")

    # Convert into categorical one-hot vectors.
    caps_y_categ = to_categorical(caps_y_pad, num_classes=10000)
    print("\tCategorical one-hot vectors done!")

    # Convert the categories into 1 or 0 vectors (True or False)
    cats_vec = []
    for caption_cats in cats:
        caption_cats_vectors = []
        for category in all_categories:
            if category in caption_cats:
                caption_cats_vectors.append(1)
            else:
                caption_cats_vectors.append(0)
        cats_vec.append(np.array(caption_cats_vectors))
    cats_vec = np.array(cats_vec)
    print("\tCategories vectors done.")
    return caps_X_pad, caps_y_categ, cats_vec


if __name__ == "__main__":
    parser = ArgumentParser("Train a model.")    
    # Add your own options as flags HERE as necessary (and some will be
    # necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the "
                             "assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed "
                             "per category. (optional)",
                        required=False)
    parser.add_argument('checkpointdir', type=str,
                        help="directory for storing checkpointed models and "
                             "other metadata (recommended to create a directory"
                             "under /scratch/)")
    parser.add_argument('modelfile', type=str, help="output model file")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in models/" + args.modelfile)
    print("Working directory at " + args.checkpointdir)
    print("Maximum instances is " + str(args.maxinstances))

    if len(args.categories) < 2:
        print("Too few categories (<2).")
        exit(0)

    print("The queried COCO categories are:")
    for c in args.categories:
        print("\t" + c)

    print("Executing option " + args.option)
    if args.option == 'A':
        optA()
    elif args.option == 'B':
        optB(args.categories, args.modelfile, args.maxinstances)
    else:
        print("Option does not exist.")
        exit(0)
