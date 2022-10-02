import numpy as np


def KL_diveregence(y, z):
    return 1/2 * (np.sum(y * np.log(y/z)) + np.sum(z * np.log(z/y)))


'''
    This function returns the intelligibility score between Z and Y.
    Arguments:
        Z: posterior features of test speech (list)
        Y: posterior features of reference speech (list)
'''
def match_score(Y, Z):
    M, N = len(Y), len(Z)
    L = np.zeros((M, N))
    L[0, 0] = KL_diveregence(Y[0], Z[0])
    for i in range(1, M):
        L[i, 0] = KL_diveregence(Y[i], Z[0]) + L[i-1, 0]
    for i in range(1, N):
        L[0, i] = KL_diveregence(Y[0], Z[i]) + L[0, i-1]
    for i in range(1, M):
        for j in range(1, N):
            L[i, j] = KL_diveregence(Y[i], Z[j]) + np.min([L[i-1, j], L[i, j-1], L[i-1, j-1]])
    r, c = M-1, N-1
    if r == 0 and c == 0:
        return L[M-1, N-1]
    path_length = 0
    while r != 0 and c != 0:
        index = np.argmin([L[i-1, j], L[i, j-1], L[i-1, j-1]])
        if index == 0:
            r = r-1
        elif index == 1:
            c = c-1
        else:
            r = r-1
            c = c-1
        path_length += 1
    path_length += (r + c)
    return L[M-1, N-1] / path_length
