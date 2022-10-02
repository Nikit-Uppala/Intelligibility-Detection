import numpy as np
from match_score import match_score


'''
    Returns whether the word is intelligible or not.
    Arguments:
    Z_w: posterior features for test speech of the word.
    Y_w: list of reference posterior features of the word.
    threshold: threshold
'''
def is_intelligible(Z_w, Y_w, threshold):
    num_votes = 0
    K = len(Y_w)
    for Y_wk in Y_w:
        score = match_score(Y_wk, Z_w)
        if score <= threshold:
            num_votes += 1
    if num_votes >= np.ceil(K / 2): return True
    return False


'''
    Returns the intelligibility score for the given speaker.
    Arguments:
    Z: list of test speech posterior features for W words.
    Y: list of (list of K reference posterior features) for W words.
    threshold: threshold 
'''
def intelligibility_score(Z, Y, threshold):
    W = len(Z)
    correct_words = 0
    for w in range(W):
        if is_intelligible(Z[w], Y[w], threshold):
            correct_words += 1
    return correct_words / W * 100
