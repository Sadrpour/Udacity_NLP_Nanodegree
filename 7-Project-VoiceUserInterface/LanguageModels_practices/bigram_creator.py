from nltk.tokenize import word_tokenize
from nltk import bigrams

test_sentences = [
    'the old man spoke to me',
    'me to spoke man old the',
    'old man me old man me',
]

def sentence_to_bigrams(sentence):
    """
    Add start '<s>' and stop '</s>' tags to the sentence and tokenize it into a list
    of lower-case words (sentence_tokens) and bigrams (sentence_bigrams)
    :param sentence: string
    :return: list, list
        sentence_tokens: ordered list of words found in the sentence
        sentence_bigrams: a list of ordered two-word tuples found in the sentence
    """
    #TODO implement
    sentence_tokens = ['<s>'] + word_tokenize(sentence) + ['</s>']
    sentence_bigrams = bigrams(sentence_tokens)
    return sentence_tokens, sentence_bigrams


def sentence_to_bigrams_simple(sentence):
    """
    Add start '<s>' and stop '</s>' tags to the sentence and tokenize it into a list
    of lower-case words (sentence_tokens) and bigrams (sentence_bigrams)
    :param sentence: string
    :return: list, list
        sentence_tokens: ordered list of words found in the sentence
        sentence_bigrams: a list of ordered two-word tuples found in the sentence
    """
    #TODO implement
    sentence_tokens = ['<s>'] + sentence.split(" ") + ['</s>']
    sentence_bigrams = list()
    for i in range(len(sentence_tokens)-1):
        sentence_bigrams.append((sentence_tokens[i],sentence_tokens[i+1]))
    return sentence_tokens, sentence_bigrams

import ngram_quiz_3.utils as utils



def log_prob_of_sentence(sentence, bigram_log_dict):
    # get the sentence bigrams
    s_tokens, s_bigrams = utils.sentence_to_bigrams(sentence)

    # add the log probabilites of the bigrams in the sentence
    total_log_prob = 0.
    for bg in s_bigrams:
        if bg in bigram_log_dict:
            total_log_prob = total_log_prob + bigram_log_dict[bg]
        else:
            total_log_prob = total_log_prob + bigram_log_dict['<unk>']
    return total_log_prob