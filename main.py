import math
import random
import pylab
import numpy
from collections import *

class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """

    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix


def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append((word, tag))
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags


def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
        else:
            viterbi[tag][0] = -1 * float('inf')

    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
            max_state = None
            for s_prime in unique_tags:
                val1= viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                curr_value = val1 + val2
                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])
            viterbi[s][t] = max_value + val3
            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence

def compute_counts(training_data: list, order: int) -> tuple:
    """
    Calculates counts of tokens, token-word pairings, and sequences of tokens.
    Inputs:
        training_data: A list of (word, POS-tag) pairs
        order: An integer denoting the order of the HMM
    Returns: A tuple containing the number of unique tokens, counts for tag-word pairs, counts for
    the frequency of each tag, counts for the number of consecutive pairs of tags, and, if the order
    is 3, the count of the number of three consecutive tags.
    """

    num_tokens = len(training_data)

    # Holds the C(t_i, w_i) dictionary
    ctw = defaultdict(lambda: defaultdict(int))
    for entry in training_data:
        tag = entry[1]
        word = entry[0]
        ctw[tag][word] += 1

    # Holds the C(t_i) dictionary
    ct = defaultdict(int)
    for entry in training_data:
        tag = entry[1]
        ct[tag] += 1

    # Holds the C(t_i-1,t_i) dictionary
    ctt = defaultdict(lambda: defaultdict(int))
    # Idx for t_i-1, goes from first index to second-to-last index
    for idx in range(len(training_data) - 1):
        # Gets the pair of tags
        prev_tag = training_data[idx][1]
        curr_tag = training_data[idx + 1][1]
        ctt[prev_tag][curr_tag] += 1

    # Holds the C(t_i-2,t_i-1,t) dictionary
    cttt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # Idx for t_i-2, goes from first index to third-to-last index
    for idx in range(len(training_data) - 2):
        # Gets the pair of tags
        prev_prev_tag = training_data[idx][1]
        prev_tag = training_data[idx + 1][1]
        curr_tag = training_data[idx + 2][1]
        cttt[prev_prev_tag][prev_tag][curr_tag] += 1
    if order == 2:
        return num_tokens, ctw, ct, ctt
    elif order == 3:
        return num_tokens, ctw, ct, ctt, cttt


def compute_initial_distribution(training_data: list, order: int) -> dict:
    """
	Computes initial probability distributions for a single tag starting a sentence (if order=2)
	or a pair of tags starting a sentence (if order=3).
	Inputs:
        training_data: A list of (word, POS-tag) pairs
        order: An integer denoting the order of the HMM
    Returns: A 1D dictionary if order is 2 mapping tags to probabilities, or a 2D dictionary if order
    is 3 mapping the first tag to the next tag then to the probability.

	"""
    if order == 2:
        # Stores the initial distribution
        initial_distribution = defaultdict(int)
        num_tags = 0
        # Add the first tag
        initial_distribution[training_data[0][1]] += 1
        num_tags += 1
        # Iterate through the rest of the data, noting tags after periods
        for idx in range(1, len(training_data)):
            # Words after periods start sentences and thus must be counted
            # However, if the period ends the file then there is no new sentence to count
            if training_data[idx][1] == "." and idx < (len(training_data) - 1):
                initial_distribution[training_data[idx + 1][1]] += 1
                num_tags += 1
        # Convert frequencies to probabilities
        for tag in initial_distribution.keys():
            initial_distribution[tag] = initial_distribution[tag] / num_tags
        return initial_distribution
    elif order == 3:
        # Stores the initial distribution
        initial_distribution = defaultdict(lambda: defaultdict(int))
        num_bigrams = 0
        # Add the first bigram
        initial_distribution[training_data[0][1]][training_data[1][1]] += 1
        num_bigrams += 1

        for idx in range(2, len(training_data)):
            if training_data[idx][1] == "." and idx < (len(training_data) - 2):
                initial_distribution[training_data[idx + 1][1]][training_data[idx + 2][1]] += 1
                num_bigrams += 1
        # Convert frequencies to probabilities
        for tag_1 in initial_distribution.keys():
            for tag_2 in initial_distribution[tag_1].keys():
                initial_distribution[tag_1][tag_2] = initial_distribution[tag_1][tag_2] / num_bigrams
        return initial_distribution


def compute_emission_probabilities(unique_words: list, unique_tags: list, W: dict, C: dict) -> dict:
    """
    Computes the emission probability for every tag emitting every word.
    Inputs:
        unique_words: A list of unique words in the training data.
        unique_tags: A list of unique tags in the training data.
        W: A 2D dictionary in which W[tag][word] is the number of times word is tagged with tag.
        C: A dictionary in which W[tag] is the number of times tag appears.
    Returns: A dictionary in which dict[tag][word] is the probability that tag emits word.
    """

    # Will hold the emission probabilities
    emit_probs = defaultdict(lambda: defaultdict(int))

    # Iterate through all unique tag, word pairs
    for tag in unique_tags:
        for word in unique_words:
            # Formula to calculate emission probability
            if W[tag][word] / C[tag] != 0:
                emit_probs[tag][word] = W[tag][word] / C[tag]

    return emit_probs


def compute_lambdas(unique_tags: list, num_tokens: int, C1: dict, C2: dict, C3: dict, order: int) -> list:
    """
    Computes the lambdas for the calculation of transition probabilities.
    Inputs:
        unique_tags: A list of all tags.
        num_tokens: The number of unique tokens.
        C1: A dictionary containing the frequencies of tag appearances.
        C2: A dictionary of bigram frequencies.
        C3: A dictionary of trigram frequencies.
        order: The order of the HMM.
    Returns: A list containing the lambda values.
    """
    if order == 3:
        # Initialize the lambdas
        l = [0, 0, 0]

        # Iterate through all t_i-2, t_i-1, t_i trigrams
        for t2 in unique_tags:
            for t1 in unique_tags:
                for t in unique_tags:
                    if C3[t2][t1][t] > 0:
                        # Calculate the alphas
                        alph = [0, 0, 0]
                        if num_tokens != 0:
                            alph[0] = (C1[t] - 1) / num_tokens
                        if (C1[t1] - 1) != 0:
                            alph[1] = (C2[t1][t] - 1) / (C1[t1] - 1)
                        if (C2[t2][t1] - 1) != 0:
                            alph[2] = (C3[t2][t1][t] - 1) / (C2[t2][t1] - 1)
                        # Find argmax i and modify its corresponding lambda
                        i = numpy.argmax(alph)
                        l[i] += C3[t2][t1][t]
        denom = sum(l)
        l = [(lam / denom) for lam in l]
        return l

    elif order == 2:
        # Initialize the lambdas
        l = [0, 0, 0]

        # Iterate through all t_i-1, t_i bigrams
        for t1 in unique_tags:
            for t in unique_tags:
                if C2[t1][t] > 0:
                    # Calculate the alphas
                    alph = [0, 0]
                    if num_tokens != 0:
                        alph[0] = (C1[t] - 1) / num_tokens
                    if (C1[t1] - 1) != 0:
                        alph[1] = (C2[t1][t] - 1) / (C1[t1] - 1)
                    # Find argmax i and modify its corresponding lambda
                    i = numpy.argmax(alph)
                    l[i] += C2[t1][t]
        denom = sum(l)
        l = [(lam / denom) for lam in l]
        return l


def build_hmm(training_data: list, unique_tags: list, unique_words: list, order: int, use_smoothing: bool):
    """
    Builds a fully trained HMM, either 2nd or 3rd other.
    Arguments:
        training_data: A list of (word, POS-tag) tuples.
        unique_tags: A list of unique tags.
        unique_words: A list of unique words.
        order: The order of the HMM
        use_smoothing: Indicates whether or not to use smoothing.
    Returns: A fully trained HMM.
    """
    # Get useful counts
    counts = compute_counts(training_data, 3)
    num_tokens = counts[0]
    ctw = counts[1]
    ct = counts[2]
    ctt = counts[3]
    cttt = counts[4]

    # Create the initial distribution
    initial_distribution = compute_initial_distribution(training_data, order)

    # Create the emission_matrix
    emission_matrix = compute_emission_probabilities(unique_words, unique_tags, ctw, ct)

    # Create the transition matrix
    transition_matrix = None
    if order == 2:
        # Calculate the lambdas based on whether smoothing is desired
        lambdas = []
        if use_smoothing:
            lambdas = compute_lambdas(unique_tags, num_tokens, ct, ctt, cttt, order)
        else:
            lambdas = [0, 1, 0]
        # Fill out the transition matrix
        transition_matrix = defaultdict(lambda: defaultdict(int))
        for prev_tag in unique_tags:
            for curr_tag in unique_tags:
                if ct[prev_tag] != 0 and num_tokens != 0:
                    transition_matrix[prev_tag][curr_tag] = (lambdas[1] * (ctt[prev_tag][curr_tag] / ct[prev_tag])) + (lambdas[0] * (ct[curr_tag] / num_tokens))

    elif order == 3:
        lambdas = []
        if use_smoothing:
            lambdas = compute_lambdas(unique_tags, num_tokens, ct, ctt, cttt, order)
        else:
            lambdas = [0, 0, 1]
        # Fill out the transition matrix
        transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for prev_prev_tag in unique_tags:
            for prev_tag in unique_tags:
                for curr_tag in unique_tags:
                    # Add conditionals if denoms are 0
                    if ctt[prev_prev_tag][prev_tag] != 0 and ct[prev_tag] != 0 and num_tokens != 0:
                        transition_matrix[prev_prev_tag][prev_tag][curr_tag] = (lambdas[2] * (cttt[prev_prev_tag][prev_tag][curr_tag] / ctt[prev_prev_tag][prev_tag])) + (lambdas[1] * (ctt[prev_tag][curr_tag] / ct[prev_tag])) + (lambdas[0] * (ct[curr_tag] / num_tokens))

    return HMM(order, initial_distribution, emission_matrix, transition_matrix)


def trigram_viterbi(hmm, sentence: list) -> list:
    """
    Performs the Viterbi Algorithm on a 2rd-order Markov Model to predict POS tags for a given sentence.
    Inputs:
        hmm: A second-order HMM.
        sentence: A list of English words.
    Returns: A list of (word, POS-tag) outputted by Viterbi.
    """
    # Initialization
    viterbi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for prev_tag in unique_tags:
        for curr_tag in unique_tags:
            if (hmm.initial_distribution[prev_tag][curr_tag] != 0) and (hmm.emission_matrix[prev_tag][sentence[0]] != 0) and (hmm.emission_matrix[curr_tag][sentence[1]] != 0):
                viterbi[prev_tag][curr_tag][1] = (math.log(hmm.initial_distribution[prev_tag][curr_tag])
                                                  + math.log(hmm.emission_matrix[prev_tag][sentence[0]])
                                                  + math.log(hmm.emission_matrix[curr_tag][sentence[1]]))
            else:
                viterbi[prev_tag][curr_tag][1] = -1 * float('inf')

    # Dynamic programming.
    for t in range(2, len(sentence)):
        backpointer["No_Path"]["No_Path"][t] = "No_Path"
        for s in unique_tags:
            for s_prime in unique_tags:
                max_value = -1 * float('inf')
                max_state = None
                for s_prime_prime in unique_tags:
                    val1 = viterbi[s_prime_prime][s_prime][t - 1]
                    val2 = -1 * float('inf')
                    if hmm.transition_matrix[s_prime_prime][s_prime][s] != 0:
                        val2 = math.log(hmm.transition_matrix[s_prime_prime][s_prime][s])
                    curr_value = val1 + val2
                    if curr_value > max_value:
                        max_value = curr_value
                        max_state = s_prime_prime
                val3 = -1 * float('inf')
                if hmm.emission_matrix[s][sentence[t]] != 0:
                    val3 = math.log(hmm.emission_matrix[s][sentence[t]])
                viterbi[s_prime][s][t] = max_value + val3
                if max_state == None:
                    backpointer[s_prime][s][t] = "No_Path"
                else:
                    backpointer[s_prime][s][t] = max_state

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    second_to_last_state = None
    final_time = len(sentence) - 1
    for s in unique_tags:
        for s_prime in unique_tags:
            if viterbi[s_prime][s][final_time] > max_value:
                max_value = viterbi[s_prime][s][final_time]
                last_state = s
                second_to_last_state = s_prime
    if last_state == None or second_to_last_state == None:
        last_state = "No_Path"
        second_to_last_state = "No Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence) - 1], last_state))
    tagged_sentence.append((sentence[len(sentence) - 2], second_to_last_state))
    for i in range(len(sentence) - 3, -1, -1):
        next_tag = tagged_sentence[-1][1]
        next_next_tag = tagged_sentence[-2][1]
        curr_tag = backpointer[next_tag][next_next_tag][i + 2]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence

############ END OF COPIED CODE ################################################################

# Read the training file and get the relevant data
training_data = read_pos_file("training.txt")
training_word_pos = training_data[0]
training_unique_words = training_data[1]
training_unique_tags = training_data[2]
# Read the tagged testing file and get the relevant data
testing_data = read_pos_file("testdata_tagged.txt")
testing_word_pos = testing_data[0]
testing_unique_words = testing_data[1]
testing_unique_tags = testing_data[2]

# Use the testing data for form a list of lists (a list of sentences) to use for testing
testing_sentences = []
curr_sentence = []
for pair in testing_word_pos:
    word = pair[0]
    if word == ".":
        curr_sentence.append(word)
        testing_sentences.append(curr_sentence)
        curr_sentence = []
    else:
        curr_sentence.append(word)

# Get the different percentages of the training dataset that the HMMs will be built on
train_1 = training_word_pos[:int(len(training_word_pos)*0.01)]
train_5 = training_word_pos[:int(len(training_word_pos)*0.05)]
train_10 = training_word_pos[:int(len(training_word_pos)*0.10)]
train_25 = training_word_pos[:int(len(training_word_pos)*0.25)]
train_50 = training_word_pos[:int(len(training_word_pos)*0.50)]
train_75 = training_word_pos[:int(len(training_word_pos)*0.75)]
train_100 = list(training_word_pos)

# Get lists of unique tags and unique words for every training dataset
unique_tags_1 = list(set([pair[1] for pair in train_1]))
unique_words_1 = list(set([pair[0] for pair in train_1]))
unique_tags_5 = list(set([pair[1] for pair in train_5]))
unique_words_5 = list(set([pair[0] for pair in train_5]))
unique_tags_10 = list(set([pair[1] for pair in train_10]))
unique_words_10 = list(set([pair[0] for pair in train_10]))
unique_tags_25 = list(set([pair[1] for pair in train_25]))
unique_words_25 = list(set([pair[0] for pair in train_25]))
unique_tags_50 = list(set([pair[1] for pair in train_50]))
unique_words_50 = list(set([pair[0] for pair in train_50]))
unique_tags_75 = list(set([pair[1] for pair in train_75]))
unique_words_75 = list(set([pair[0] for pair in train_75]))
unique_tags_100 = list(set([pair[1] for pair in train_100]))
unique_words_100 = list(set([pair[0] for pair in train_100]))


def update_hmm(unique_training_words, unique_testing_words, hmm):
	"""
    Updates a given HMM so that it has a very small emission probability for unknown words in the testing data,
    while also normalizing emission probabilities for every tag.
    Inputs:
        unique_training_words: A list of words the HMM was trained on.
        unique_testing_words: A list of words the HMM will be tested on.
        hmm: A given HMM.
    Modifies: The HMM.
	"""
	words = unique_testing_words
	new_words = []
	epsilon = 0.00001
	count = 0
	for word in words:
		if word not in unique_training_words:
			new_words.append(word)
			for tag in hmm.emission_matrix.keys():
				count += 1
				hmm.emission_matrix[tag][word] = 0
	for tag in hmm.emission_matrix.keys():
		for word in hmm.emission_matrix[tag].keys():
			hmm.emission_matrix[tag][word] += epsilon
			if word not in new_words:
				hmm.emission_matrix[tag][word] /= (1 + (epsilon * count))

# Write a function to account for unknown words (give them an emission probability of 0.00001) (OLD FUNCTION)
# def update_hmm2(training_words, testing_words, hmm):
#     """
#     Updates a given HMM so that it has a very small emission probability for unknown words in the testing data,
#     while also normalizing emission probabilities for every tag.
#     Inputs:
#         training_words: A list of words the HMM was trained on.
#         testing_words: A list of words the HMM will be tested on.
#         hmm: A given HMM.
#     Modifies: The HMM.
#     """
#     for test_word in testing_words:
#         # Give unknown testing words very small emission probability
#         if test_word not in training_words:
#             for tag in hmm.emission_matrix.keys():
#                 hmm.emission_matrix[tag][test_word] = 0.00001
#         # Give known testing words an increase of epsilon
#         else:
#             for tag in hmm.emission_matrix.keys():
#                 hmm.emission_matrix[tag][test_word] += 0.00001
#
#     # Normalize the probabilities for each tag to 1
#     for tag in hmm.emission_matrix.keys():
#         total_prob = sum(hmm.emission_matrix[tag].values())
#         for word in hmm.emission_matrix[tag].keys():
#             hmm.emission_matrix[tag][word] = hmm.emission_matrix[tag][word] / total_prob

# Write a function to run experiments on an HMM

def run_experiment(hmm, sentences, order):
    """
    Runs the Viterbi algorithm on a given HMM to produce the predicted tags for a given amount of text.
    Inputs:
        hmm: An HMM, bigram or trigram.
        sentences: A list of lists, each list being a list of words that constitute a sentence.
        order: The order of the HMM.
    Returns: A list of predicted tags generated by the Viterbi algorithms in the same order as the words.
    """
    # Will hold the tagged output
    tagged = []
    if order == 2:
        for sentence in sentences:
            tagged.append(bigram_viterbi(hmm, sentence))
    elif order == 3:
        for sentence in sentences:
            tagged.append(trigram_viterbi(hmm, sentence))
    # Turns the lists of lists into a single merged list
    tagged = sum(tagged, [])

    return tagged

# Create a function to measure accuracy values
def compute_accuracy(predicted_tagging, actual_tagging):
    """
    Computes the accuracy of a list of predicted tags related to the actual tags.
    Inputs:
        predicted_tagging: A list of predicted tags.
        actual_tagging: A list of the actual tags for the test data.
    Returns: The percentage of tags that were correctly predicted.
    """
    correct = 0
    total = len(actual_tagging)
    if len(actual_tagging) == len(predicted_tagging):
        for idx in range(len(actual_tagging)):
            if predicted_tagging[idx][1] == actual_tagging[idx][1]:
                correct += 1
    else:
        print("ERROR: DATA LENGTHS DIFFERENT")

    return 100 * (correct / total)

#### EXPERIMENT 1 ####################################################################################################

# Create the bigram HMMs without smoothing
bi_unsmooth_hmm_1 = build_hmm(train_1, unique_tags_1, unique_words_1, 2, False)
bi_unsmooth_hmm_5 = build_hmm(train_5, unique_tags_5, unique_words_5, 2, False)
bi_unsmooth_hmm_10 = build_hmm(train_10, unique_tags_10, unique_words_10, 2, False)
bi_unsmooth_hmm_25 = build_hmm(train_25, unique_tags_25, unique_words_25, 2, False)
bi_unsmooth_hmm_50 = build_hmm(train_50, unique_tags_50, unique_words_50, 2, False)
bi_unsmooth_hmm_75 = build_hmm(train_75, unique_tags_75, unique_words_75, 2, False)
bi_unsmooth_hmm_100 = build_hmm(train_100, unique_tags_100, unique_words_100, 2, False)


# Update the HMMs
update_hmm(unique_words_1, testing_unique_words, bi_unsmooth_hmm_1)
update_hmm(unique_words_5, testing_unique_words, bi_unsmooth_hmm_5)
update_hmm(unique_words_10, testing_unique_words, bi_unsmooth_hmm_10)
update_hmm(unique_words_25, testing_unique_words, bi_unsmooth_hmm_25)
update_hmm(unique_words_50, testing_unique_words, bi_unsmooth_hmm_50)
update_hmm(unique_words_75, testing_unique_words, bi_unsmooth_hmm_75)
update_hmm(unique_words_100, testing_unique_words, bi_unsmooth_hmm_100)


# Run the experiments and obtain the accuracy values
bus_tagged1 = run_experiment(bi_unsmooth_hmm_1, testing_sentences, 2)
print("BIGRAM NO SMOOTHING 1% ACCURACY: ", compute_accuracy(bus_tagged1, testing_word_pos))
bus1_data = compute_accuracy(bus_tagged1, testing_word_pos)
bus_tagged5 = run_experiment(bi_unsmooth_hmm_5, testing_sentences, 2)
print("BIGRAM NO SMOOTHING 5% ACCURACY: ", compute_accuracy(bus_tagged5, testing_word_pos))
bus5_data = compute_accuracy(bus_tagged5, testing_word_pos)
bus_tagged10 = run_experiment(bi_unsmooth_hmm_10, testing_sentences, 2)
print("BIGRAM NO SMOOTHING 10% ACCURACY: ", compute_accuracy(bus_tagged10, testing_word_pos))
bus10_data = compute_accuracy(bus_tagged10, testing_word_pos)
bus_tagged25 = run_experiment(bi_unsmooth_hmm_25, testing_sentences, 2)
print("BIGRAM NO SMOOTHING 25% ACCURACY: ", compute_accuracy(bus_tagged25, testing_word_pos))
bus25_data = compute_accuracy(bus_tagged25, testing_word_pos)
bus_tagged50 = run_experiment(bi_unsmooth_hmm_50, testing_sentences, 2)
print("BIGRAM NO SMOOTHING 50% ACCURACY: ", compute_accuracy(bus_tagged50, testing_word_pos))
bus50_data = compute_accuracy(bus_tagged50, testing_word_pos)
bus_tagged75 = run_experiment(bi_unsmooth_hmm_75, testing_sentences, 2)
print("BIGRAM NO SMOOTHING 75% ACCURACY: ", compute_accuracy(bus_tagged75, testing_word_pos))
bus75_data = compute_accuracy(bus_tagged75, testing_word_pos)
bus_tagged100 = run_experiment(bi_unsmooth_hmm_100, testing_sentences, 2)
print("BIGRAM NO SMOOTHING 100% ACCURACY: ",compute_accuracy(bus_tagged100, testing_word_pos))
bus100_data = compute_accuracy(bus_tagged100, testing_word_pos)
#### EXPERIMENT 3 ####################################################################################################

# Create the bigram HMMs with smoothing
bi_smooth_hmm_1 = build_hmm(train_1, unique_tags_1, unique_words_1, 2, True)
bi_smooth_hmm_5 = build_hmm(train_5, unique_tags_5, unique_words_5, 2, True)
bi_smooth_hmm_10 = build_hmm(train_10, unique_tags_10, unique_words_10, 2, True)
bi_smooth_hmm_25 = build_hmm(train_25, unique_tags_25, unique_words_25, 2, True)
bi_smooth_hmm_50 = build_hmm(train_50, unique_tags_50, unique_words_50, 2, True)
bi_smooth_hmm_75 = build_hmm(train_75, unique_tags_75, unique_words_75, 2, True)
bi_smooth_hmm_100 = build_hmm(train_100, unique_tags_100, unique_words_100, 2, True)

# Update the HMMs
update_hmm(unique_words_1, testing_unique_words, bi_smooth_hmm_1)
update_hmm(unique_words_5, testing_unique_words, bi_smooth_hmm_5)
update_hmm(unique_words_10, testing_unique_words, bi_smooth_hmm_10)
update_hmm(unique_words_25, testing_unique_words, bi_smooth_hmm_25)
update_hmm(unique_words_50, testing_unique_words, bi_smooth_hmm_50)
update_hmm(unique_words_75, testing_unique_words, bi_smooth_hmm_75)
update_hmm(unique_words_100, testing_unique_words, bi_smooth_hmm_100)


# Run the experiments and obtain the accuracy values
bs_tagged1 = run_experiment(bi_smooth_hmm_1, testing_sentences, 2)
print("BIGRAM WITH SMOOTHING 1% ACCURACY: ", compute_accuracy(bs_tagged1, testing_word_pos))
bs1_data = compute_accuracy(bs_tagged1, testing_word_pos)
bs_tagged5 = run_experiment(bi_smooth_hmm_5, testing_sentences, 2)
print("BIGRAM WITH SMOOTHING 5% ACCURACY: ", compute_accuracy(bs_tagged5, testing_word_pos))
bs5_data = compute_accuracy(bs_tagged5, testing_word_pos)
bs_tagged10 = run_experiment(bi_smooth_hmm_10, testing_sentences, 2)
print("BIGRAM WITH SMOOTHING 10% ACCURACY: ", compute_accuracy(bs_tagged10, testing_word_pos))
bs10_data = compute_accuracy(bs_tagged10, testing_word_pos)
bs_tagged25 = run_experiment(bi_smooth_hmm_25, testing_sentences, 2)
print("BIGRAM WITH SMOOTHING 25% ACCURACY: ", compute_accuracy(bs_tagged25, testing_word_pos))
bs25_data = compute_accuracy(bs_tagged25, testing_word_pos)
bs_tagged50 = run_experiment(bi_smooth_hmm_50, testing_sentences, 2)
print("BIGRAM WITH SMOOTHING 50% ACCURACY: ", compute_accuracy(bs_tagged50, testing_word_pos))
bs50_data = compute_accuracy(bs_tagged50, testing_word_pos)
bs_tagged75 = run_experiment(bi_smooth_hmm_75, testing_sentences, 2)
print("BIGRAM WITH SMOOTHING 75% ACCURACY: ", compute_accuracy(bs_tagged75, testing_word_pos))
bs75_data = compute_accuracy(bs_tagged75, testing_word_pos)
bs_tagged100 = run_experiment(bi_smooth_hmm_100, testing_sentences, 2)
print("BIGRAM WITH SMOOTHING 100% ACCURACY: ", compute_accuracy(bs_tagged100, testing_word_pos))
bs100_data = compute_accuracy(bs_tagged100, testing_word_pos)

#### EXPERIMENT 2 ####################################################################################################

# Create the trigram HMMs without smoothing
tri_unsmooth_hmm_1 = build_hmm(train_1, unique_tags_1, unique_words_1, 3, False)
tri_unsmooth_hmm_5 = build_hmm(train_5, unique_tags_5, unique_words_5, 3, False)
tri_unsmooth_hmm_10 = build_hmm(train_10, unique_tags_10, unique_words_10, 3, False)
tri_unsmooth_hmm_25 = build_hmm(train_25, unique_tags_25, unique_words_25, 3, False)
tri_unsmooth_hmm_50 = build_hmm(train_50, unique_tags_50, unique_words_50, 3, False)
tri_unsmooth_hmm_75 = build_hmm(train_75, unique_tags_75, unique_words_75, 3, False)
tri_unsmooth_hmm_100 = build_hmm(train_100, unique_tags_100, unique_words_100, 3, False)


# Update the HMMs
update_hmm(unique_words_1, testing_unique_words, tri_unsmooth_hmm_1)
update_hmm(unique_words_5, testing_unique_words, tri_unsmooth_hmm_5)
update_hmm(unique_words_10, testing_unique_words, tri_unsmooth_hmm_10)
update_hmm(unique_words_25, testing_unique_words, tri_unsmooth_hmm_25)
update_hmm(unique_words_50, testing_unique_words, tri_unsmooth_hmm_50)
update_hmm(unique_words_75, testing_unique_words, tri_unsmooth_hmm_75)
update_hmm(unique_words_100, testing_unique_words, tri_unsmooth_hmm_100)


# Run the experiments and obtain the accuracy values
tus_tagged1 = run_experiment(tri_unsmooth_hmm_1, testing_sentences, 3)
print("TRIGRAM NO SMOOTHING 1% ACCURACY: ", compute_accuracy(tus_tagged1, testing_word_pos))
tus1_data = compute_accuracy(tus_tagged1, testing_word_pos)
tus_tagged5 = run_experiment(tri_unsmooth_hmm_5, testing_sentences, 3)
print("TRIGRAM NO SMOOTHING 5% ACCURACY: ", compute_accuracy(tus_tagged5, testing_word_pos))
tus5_data = compute_accuracy(tus_tagged5, testing_word_pos)
tus_tagged10 = run_experiment(tri_unsmooth_hmm_10, testing_sentences, 3)
print("TRIGRAM NO SMOOTHING 10% ACCURACY: ", compute_accuracy(tus_tagged10, testing_word_pos))
tus10_data = compute_accuracy(tus_tagged10, testing_word_pos)
tus_tagged25 = run_experiment(tri_unsmooth_hmm_25, testing_sentences, 3)
print("TRIGRAM NO SMOOTHING 25% ACCURACY: ", compute_accuracy(tus_tagged25, testing_word_pos))
tus25_data = compute_accuracy(tus_tagged25, testing_word_pos)
tus_tagged50 = run_experiment(tri_unsmooth_hmm_50, testing_sentences, 3)
print("TRIGRAM NO SMOOTHING 50% ACCURACY: ", compute_accuracy(tus_tagged50, testing_word_pos))
tus50_data = compute_accuracy(tus_tagged50, testing_word_pos)
tus_tagged75 = run_experiment(tri_unsmooth_hmm_75, testing_sentences, 3)
print("TRIGRAM NO SMOOTHING 75% ACCURACY: ", compute_accuracy(tus_tagged75, testing_word_pos))
tus75_data = compute_accuracy(tus_tagged75, testing_word_pos)
tus_tagged100 = run_experiment(tri_unsmooth_hmm_100, testing_sentences, 3)
print("TRIGRAM NO SMOOTHING 100% ACCURACY: ", compute_accuracy(tus_tagged100, testing_word_pos))
tus100_data = compute_accuracy(tus_tagged100, testing_word_pos)
#### EXPERIMENT 4 ####################################################################################################

# Create the trigram HMMs with smoothing
tri_smooth_hmm_1 = build_hmm(train_1, unique_tags_1, unique_words_1, 3, True)
tri_smooth_hmm_5 = build_hmm(train_5, unique_tags_5, unique_words_5, 3, True)
tri_smooth_hmm_10 = build_hmm(train_10, unique_tags_10, unique_words_10, 3, True)
tri_smooth_hmm_25 = build_hmm(train_25, unique_tags_25, unique_words_25, 3, True)
tri_smooth_hmm_50 = build_hmm(train_50, unique_tags_50, unique_words_50, 3, True)
tri_smooth_hmm_75 = build_hmm(train_75, unique_tags_75, unique_words_75, 3, True)
tri_smooth_hmm_100 = build_hmm(train_100, unique_tags_100, unique_words_100, 3, True)

# Update the HMMs
update_hmm(unique_words_1, testing_unique_words, tri_smooth_hmm_1)
update_hmm(unique_words_5, testing_unique_words, tri_smooth_hmm_5)
update_hmm(unique_words_10, testing_unique_words, tri_smooth_hmm_10)
update_hmm(unique_words_25, testing_unique_words, tri_smooth_hmm_25)
update_hmm(unique_words_50, testing_unique_words, tri_smooth_hmm_50)
update_hmm(unique_words_75, testing_unique_words, tri_smooth_hmm_75)
update_hmm(unique_words_100, testing_unique_words, tri_smooth_hmm_100)


# Run the experiments and obtain the accuracy values
ts_tagged1 = run_experiment(tri_smooth_hmm_1, testing_sentences, 3)
print("TRIGRAM WITH SMOOTHING 1% ACCURACY: ", compute_accuracy(ts_tagged1, testing_word_pos))
ts_tagged5 = run_experiment(tri_smooth_hmm_5, testing_sentences, 3)
print("TRIGRAM WITH SMOOTHING 5% ACCURACY: ", compute_accuracy(ts_tagged5, testing_word_pos))
ts_tagged10 = run_experiment(tri_smooth_hmm_10, testing_sentences, 3)
print("TRIGRAM WITH SMOOTHING 10% ACCURACY: ", compute_accuracy(ts_tagged10, testing_word_pos))
ts_tagged25 = run_experiment(tri_smooth_hmm_25, testing_sentences, 3)
print("TRIGRAM WITH SMOOTHING 25% ACCURACY: ", compute_accuracy(ts_tagged25, testing_word_pos))
ts_tagged50 = run_experiment(tri_smooth_hmm_50, testing_sentences, 3)
print("TRIGRAM WITH SMOOTHING 50% ACCURACY: ", compute_accuracy(ts_tagged50, testing_word_pos))
ts_tagged75 = run_experiment(tri_smooth_hmm_75, testing_sentences, 3)
print("TRIGRAM WITH SMOOTHING 75% ACCURACY: ", compute_accuracy(ts_tagged75, testing_word_pos))
ts_tagged100 = run_experiment(tri_smooth_hmm_100, testing_sentences, 3)
print("TRIGRAM WITH SMOOTHING 100% ACCURACY: ", compute_accuracy(ts_tagged100, testing_word_pos))

# Accuracy values were manually recorded to speed up running time of code, but these are the values outputted by the above
# code. The dictionaries record accuracy values for bigrams without smoothing, then trigrams without smoothing, then
# bigrams with smoothing, then trigrams with smoothing.
graph_data = [{1: 66.32443531827515, 5: 78.74743326488706, 10: 87.98767967145791, 25: 93.83983572895276, 50: 95.48254620123203, 75: 96.61190965092402, 100: 96.91991786447639},
              {1: 35.318275154004105, 5: 69.60985626283367, 10: 78.85010266940452, 25: 87.26899383983573, 50: 92.71047227926078, 75: 93.53182751540041, 100: 94.25051334702259},
              {1: 77.51540041067761, 5: 88.19301848049281, 10: 90.75975359342917, 25: 93.53182751540041, 50: 95.17453798767967, 75: 96.40657084188912, 100: 96.91991786447639},
              {1: 67.14579055441479, 5: 78.95277207392198, 10: 88.80903490759754, 25: 94.35318275154005, 50: 96.09856262833677, 75: 97.02258726899385, 100: 97.5359342915811}]

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals
def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)
def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments:
    data     -- a list of dictionaries, each of which will be plotted
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    # ymin = min(0, min(mins))
    ymin = 30
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

plot_lines(graph_data, "Accuracy of HMM POS-Tagging Over Varying Training Datasets", "Percentage of Training Data",
           "Percentage of Accurately-tagged Words", ["Bigram (no smoothing)", "Trigram( no smoothing)", "Bigram (smoothing)", "Trigram (smoothing)"])

