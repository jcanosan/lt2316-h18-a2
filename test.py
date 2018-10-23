# This is the main testing script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from argparse import ArgumentParser
import mycoco
import train
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('test')
    print("Option A not implemented!")


# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB(categories, modelfile, maxinstances, caption):
    mycoco.setmode('test')

    # Extract a list with all the categories.
    all_cats_ids = mycoco.annotcoco.loadCats(mycoco.annotcoco.getCatIds())
    all_categories = [cat['name'] for cat in all_cats_ids]

    # Collect the captions used for testing
    print("Collecting captions and categories for evaluation...")
    test_caps, test_cats = train.collect_captions_cats(categories, maxinstances)
    print("Done!")

    # Prepare the test data
    print("Preparing the captions and categories for modeling...")
    # Open the tokenizer indexes saved when running train.py
    with open('tokenizer/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    caps_X_pad, caps_y_categ, cats_vec = \
        train.prepare_data(tokenizer, all_categories, test_caps, test_cats)
    print("Done!\n")

    # Load the model
    model = load_model("models/" + modelfile)

    # Evaluate and print metrics values
    print("\nMetrics values:")
    evaluation = model.evaluate([caps_X_pad], [caps_y_categ, cats_vec])
    metrics = ["Loss", "Last word (y) loss (categorical cross-entropy)",
               "Multiclass category loss (binary cross-entropy)",
               "Last word (y) accuracy", "Multiclass category accuracy"]
    for name, value in zip(metrics, evaluation):
        print("%s = %f"%(name, value))

    # Predict the next word and the category for a specified string
    print("\nPrediction:")
    test_X_seqs = tokenizer.texts_to_sequences([caption.split(' ')])
    test_X_pad = pad_sequences(test_X_seqs, maxlen=15)
    prediction = model.predict(test_X_pad)

    # Split the two outputs
    results_caps_pred = prediction[0]
    results_cats_pred = prediction[1]

    # Decode index of the predicted last word with best score
    results_caps = np.argmax(results_caps_pred[0][-1])
    # Reverse the caption index to convert the index to the respective words
    reverse_caps_map = dict(map(reversed, tokenizer.word_index.items()))
    # Convert sequences to words
    last_wd_pred = reverse_caps_map[results_caps]

    # Decode categories
    cats_pred = [x for x in sorted(zip(results_cats_pred[0], all_categories), reverse=True)]

    # Print results
    print("Input:", caption)
    print("Last word:", last_wd_pred)
    print("Top 5 categories with scores:")
    for i, j in cats_pred[:5]:
        print("\t%s: %f" % (j, i))


if __name__ == "__main__":
    parser = ArgumentParser("Evaluate a model.")    
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
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    parser.add_argument('caption', type=str, help="an incomplete caption to "
                                                  "predict the last word and "
                                                  "categories")
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
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
        optB(args.categories, args.modelfile, args.maxinstances, args.caption)
    else:
        print("Option does not exist.")
        exit(0)
