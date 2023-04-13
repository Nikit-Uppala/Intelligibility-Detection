import numpy as np
from match_score import match_score


def is_intelligible(Z_w, Y_w, threshold, version=1):
    """Returns whether the spoken sentence is intelligible or not.

    Args:
        Z_w (array): posterior features for test speech of the word.
        Y_w (array): list of reference posterior features of the word.
        threshold (int): threshold
        version (int, optional): It can take 4 values, which are ,abs value, +ve values etc, . Defaults to 1. 
        Refer KL_diveregence function in match score for more information.

    Returns:
        Boolean: Intelligibity
    """
    num_votes = 0
    K = len(Y_w)
    for Y_wk in Y_w:
        score = match_score(Y_wk, Z_w, version)
        if score <= threshold:
            num_votes += 1
    # print(num_votes)
    # print(K)
    if num_votes >= np.ceil(K / 2): return True
    return False

#NOTE below function is depreciated
# def intelligibility_score(Z, Y, threshold, version=1):
#     """Returns the intelligibility score for the given speaker.

#     Args:
#         Z (array): list of test speech posterior features for W words.
#         Y (array): list of (list of K reference posterior features) for W words.
#         threshold (int): threshold
#         version (int, optional): It can take 4 values, which are ,abs value, +ve values etc, . Defaults to 1. 
#         Refer KL_diveregence function in match score for more information.

#     Returns:
#         float : Calculated Intelligibility Score
#     """
#     W = len(Z)
#     correct_words = 0
#     for w in range(W):
#         if is_intelligible(Z[w], Y[w], threshold, version=1):
#             correct_words += 1
#     return correct_words / W * 100
