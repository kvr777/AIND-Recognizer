import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, words: dict, hwords: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=None, verbose=False):
        self.words = words
        self.hwords = hwords
        self.sequences = words[this_word]
        self.X, self.lengths = hwords[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        BIC = float('Inf')
        best_model = None
        fitted_model = None
        for Nb_states in range(self.min_n_components, self.max_n_components +1):
            try:
                fitted_model = GaussianHMM(n_components=Nb_states, covariance_type="diag", n_iter=1000,
                                       random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                LogL = fitted_model.score(self.X, self.lengths)
                N = len(self.X)
                NDimensions = self.X.shape[1]

                p = (Nb_states-1) + (Nb_states * NDimensions ) + Nb_states * NDimensions * (NDimensions+1)/2. #for full covariance matrices

                BIC_temp = -2 * LogL + p * np.log(N)

            except:
                BIC_temp = float('Inf')

            if BIC_temp<BIC:
                BIC=BIC_temp
                best_model= fitted_model

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        # TODO implement model selection based on DIC scores
        # warnings.filterwarnings("ignore", category=RuntimeWarning)

        DIC = -float('Inf')
        DIC_Temp = -float('Inf')
        best_model = None
        fitted_model = None
        for Nb_states in range(self.min_n_components, self.max_n_components +1):
            M = 1.
            temp_model = None
            try:
                fitted_model = GaussianHMM(n_components=Nb_states, covariance_type="diag", n_iter=1000,
                                       random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                DIC_first_Log = fitted_model.score(self.X, self.lengths)
                temp_model = fitted_model
            except:
                continue

            TotalSumLog = 0

            for hword in self.hwords.keys():
                if hword != self.this_word:
                    other_X, other_length = self.hwords[hword]
                    try:
                        SumLogOtherWord = temp_model.score(other_X, other_length)
                        M+=1
                    except:
                        SumLogOtherWord =0
                    TotalSumLog += SumLogOtherWord

            if M==1:
                M= floaT('inf')
            DIC_temp = DIC_first_Log - (1/(M-1))*TotalSumLog*1.

            if DIC_temp>DIC:
                DIC=DIC_temp
                best_model= temp_model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        if len(self.lengths) < 2:
            return print("Number of samples is less than minimal number of kfolds")

        split_method = KFold(n_splits = min(len(self.lengths),3))
        BestAvgLL = -float('Inf')
        best_model = None
        temp_model = None
        for Nb_states in range(self.min_n_components, self.max_n_components + 1):
            TotalLL = 0
            CountLL = 1
            fitted_model = None
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                X_train, X_test = [],[]
                for ii in cv_train_idx:
                    X_train += self.sequences[ii]

                for yy in cv_test_idx:
                    X_test += self.sequences[yy]

                X_train, X_test = np.array(X_train), np.array(X_test)
                len_train, len_test = np.array(self.lengths)[cv_train_idx], np.array(self.lengths)[cv_test_idx]

                try:
                    fitted_model = GaussianHMM(n_components=Nb_states, covariance_type="diag", n_iter=1000,
                                               random_state=self.random_state, verbose=False).fit(X_train, len_train)

                    LogL = fitted_model.score(X_test, len_test)

                    CountLL += 1
                except:
                    LogL = 0

                TotalLL += LogL

            AvgTempLL = TotalLL/(CountLL*1.0)

            if AvgTempLL>BestAvgLL:
                BestAvgLL = AvgTempLL
                best_model = fitted_model

        return best_model