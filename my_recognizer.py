import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set
   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    #for each word from test set, extract (X, lengths) tuple
    for word_id,_ in test_set.get_all_Xlengths().items():
        prob_dict = dict()
        X_test, len_test = test_set.get_item_Xlengths(word_id)
        for word_model, model in models.items():
            # Calclulate LogLvalue for each word from dict of trained model, if cannot calculate score,
            # for this word is no probablity to be added in list
            try:
                LogL = model.score(X_test, len_test)
            except:
                LogL = -float('inf')
            prob_dict[word_model] = LogL

        # add dict with probablities for word into probablities dict.
        # find the word with the highest probability to guesses list
        probabilities.append(prob_dict)
        guesses.append(max(prob_dict, key=prob_dict.get))

    return probabilities, guesses