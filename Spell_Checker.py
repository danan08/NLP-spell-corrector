import string
from collections import defaultdict
import math
import re
import random
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self,  lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.errors_table = None

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm


    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.errors_table = error_tables

    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words
    
           Args:
               text (str): Text to evaluate.
    
           Returns:
               Float. The float should reflect the (log) probability.
        """

        return self.lm.evaluate_text(text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        if not self.errors_table or not self.lm:
            raise ValueError("Language model and error tables must be provided before spell checking.")
        sentences = sent_tokenize(text)
        corrected_sentences = [self.correct_sentence(sentence, alpha) for sentence in sentences]
        return ' '.join(corrected_sentences)

    def damerau_levenshtein_distance_recursive(a, b, i, j, memo):
        if min(i, j) < 0:
            return max(i, j) + 1

        if (i, j) in memo:
            return memo[(i, j)]

        if a[i] == b[j]:
            cost = 0
        else:
            cost = 1

        d = min(
            a.damerau_levenshtein_distance_recursive(a, b, i - 1, j, memo) + 1,
            a.damerau_levenshtein_distance_recursive(a, b, i, j - 1, memo) + 1,
            a.damerau_levenshtein_distance_recursive(a, b, i - 1, j - 1, memo) + cost
        )

        if i > 0 and j > 0 and a[i] == b[j - 1] and a[i - 1] == b[j]:
            d = min(
                d,
                a.damerau_levenshtein_distance_recursive(a, b, i - 2, j - 2, memo) + cost
            )

        memo[(i, j)] = d
        return d

    def generate_candidates(self, word):
        vocabulary = self.lm.get_vocabulary()
        candidates = {word}  # Include the original word as a candidate

        # Generate candidates with one edit distance
        for i in range(len(word) + 1):
            for letter in self.lm.alphabet:
                # Insertion
                candidate = word[:i] + letter + word[i:]
                if candidate in vocabulary:
                    candidates.add(candidate)

                # Substitution
                if i < len(word):
                    candidate = word[:i] + letter + word[i + 1:]
                    if candidate in vocabulary:
                        candidates.add(candidate)

            # Generate candidates with transposition
            if i < len(word) - 1:
                candidate = word[:i] + word[i + 1] + word[i] + word[i + 2:]
                if candidate in vocabulary:
                    candidates.add(candidate)

            # Generate candidates with deletion
            candidate = word[:i] + word[i + 1:]
            if candidate in vocabulary:
                candidates.add(candidate)

        return candidates

    def correct(self, word, prev_word, next_word, alpha):
        candidates = self.generate_candidates(word)
        scored_candidates = self.score_candidates(candidates, word, prev_word, next_word)

        if scored_candidates:
            return max(scored_candidates, key=scored_candidates.get)
        else:
            return word

    def correct_sentence(self, sentence, alpha):
        words = word_tokenize(sentence)
        corrected_words = []
        corrected = False  # Flag to indicate if a word has already been corrected in the sentence
        for i, word in enumerate(words):
            if i == 0:
                prev_word = None
            else:
                prev_word = words[i - 1]

            if i == len(words) - 1:
                next_word = None
            else:
                next_word = words[i + 1]

            if not corrected:  # Only correct the first misspelled word in the sentence
                scored_candidates = self.score_candidates(self.generate_candidates(word), word, prev_word, next_word)

                if scored_candidates:
                    best_candidate = max(scored_candidates, key=scored_candidates.get)
                    if best_candidate != word:
                        corrected_words.append(best_candidate)
                        corrected = True
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def score_candidates(self, candidates, original_word, prev_word, next_word):
        scores = {}
        for candidate in candidates:
            if len(candidate) == 0 or len(original_word) == 0:
                continue

            last_char_candidate = candidate[-1]
            last_char_original = original_word[-1]

            insertion_key = last_char_original + last_char_candidate
            insertion_error = self.errors_table["insertion"].get(insertion_key, 0)

            deletion_key = last_char_candidate + last_char_original
            deletion_error = self.errors_table["deletion"].get(deletion_key, 0)

            substitution_key = last_char_original + last_char_candidate
            substitution_error = self.errors_table["substitution"].get(substitution_key, 0)

            transposition_key = last_char_candidate + last_char_original
            transposition_error = self.errors_table["transposition"].get(transposition_key, 0)

            error_sum = insertion_error + deletion_error + substitution_error + transposition_error
            if error_sum == 0:
                continue

            p_x_given_w = math.log((insertion_error / error_sum) + 1e-10) + \
                          math.log((deletion_error / error_sum) + 1e-10) + \
                          math.log((substitution_error / error_sum) + 1e-10) + \
                          math.log((transposition_error / error_sum) + 1e-10)

            if prev_word and next_word:
                ngram = (prev_word, candidate, next_word)
                p_w_given_w2_w1 = math.log(self.lm.evaluate_text(ngram) + 1e-10)
            else:
                p_w_given_w2_w1 = 0

            scores[candidate] = p_x_given_w + p_w_given_w2_w1

        return scores


    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.model_dict = None #a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            #NOTE: This dictionary format is inefficient and insufficient (why?), therefore  you can (even encouraged to)
            # use a better data structure.
            # However, you are requested to support this format for two reasons:
            # (1) It is very straight forward and force you to understand the logic behind LM, and
            # (2) It serves as the normal form for the LM so we can call get_model_dictionary() and peek into you model.
            self.n = n
            self.ngrams = defaultdict(int)
            self.word_frequencies = defaultdict(int)
            self.ngram_frequencies = defaultdict(int)
            self.vocabulary = set()
            self.alphabet = string.ascii_lowercase + string.digits + string.punctuation
            self.contexts_d = defaultdict(lambda: defaultdict(int))


        def build_model(self, text):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            #init the class variables
            sentences = sent_tokenize(text)
            for sentence in sentences:
                self.add_sentence(sentence)
            tokens = nltk.word_tokenize(text)
            ngrams = list(nltk.ngrams(tokens, self.n, pad_left=True, pad_right=True))
            for ngram in ngrams:
                self.ngram_frequencies[ngram] += 1
                self.ngrams[ngram[:-1]] += 1
                self.contexts_d[ngram[:-1]][ngram[-1]] += 1
            ngrams = defaultdict(int)
            tokens = text.lower().split()
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                ngrams[ngram] += 1
            self.model_dict= ngrams

        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.

            """

            if context is None:
                context = random.choice(list(self.contexts_d.keys()))
            context_str = ' '.join(context)
            generated_text = list(context_str.split())
            while len(generated_text) < n:
                next_word = self.generate_next_word(context)
                if next_word is None:
                    break
                generated_text.append(next_word)
                context = tuple(generated_text[-len(context):])
            return ' '.join(generated_text)

        def generate_next_word(self, context_str):
            if context_str not in self.contexts_d:
                return None
            candidates = list(self.contexts_d[context_str].keys())
            probabilities = [self.get_ngram_probability(context_str + (candidate,)) for candidate in candidates]
            return random.choices(candidates, weights=probabilities, k=1)[0]

        def get_ngram_probability(self, ngram):
            if ngram not in self.ngram_frequencies:
                return 0
            context = ngram[:-1]
            return self.ngram_frequencies[ngram] / self.ngrams[context]

        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   Float. The float should reflect the (log) probability.
            """
            ngram = text
            if len(ngram) != self.n:
                raise ValueError("The ngram length does not match the language model's n.")
            context = ngram[:-1]
            if context not in self.ngrams:
                return 0
            return self.ngram_frequencies[ngram] / self.ngrams[context]

        def add_sentence(self, sentence):
            words = sentence.split()
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i + self.n])
                self.ngrams[ngram] += 1
                self.word_frequencies[ngram[-1]] += 1
                self.ngram_frequencies[ngram] += 1
                if i == 0:
                    self.word_frequencies[ngram[0]] += 1
                self.vocabulary.add(ngram[-1])

        def get_vocabulary(self):
            return self.vocabulary

        def get_context_d(self,text):
            words = text.split()
            ngrams = defaultdict(int)
            contexts = defaultdict(int)
            contexts_d = defaultdict(lambda: defaultdict(int))
            for i in range(len(words) - self.n + 1):
                ngrams[' '.join(words[i:i + self.n])] += 1
                contexts[' '.join(words[i:i + self.n - 1])] += 1
                contexts_d[' '.join(words[i:i + self.n - 1])][words[i + self.n - 1]] += 1
            return contexts_d

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed

                Returns:
                    float. The smoothed probability.
            """
            vocab_size = len(self.vocabulary)
            count_ngram = self.ngram_frequencies[ngram]
            count_n_minus_1_gram = self.ngrams[ngram[:-1]]
            return (count_ngram + 1) / (count_n_minus_1_gram + vocab_size)

def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Dana Nehemia', 'id': '313548810', 'email': 'dananeh@post.bgu.ac.il'}