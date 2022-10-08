import numpy as np


def KL_diveregence(y, z, version):
    if version == 1: # taking abs value
        y_pos = np.abs(y)
        z_pos = np.abs(z)
    elif version == 2: # taking only positive values
        positives = (y >= 0) & (z >= 0)
        y_pos = y[positives]
        z_pos = z[positives]
    elif version == 3: # replacing negative values with small value
        eps = 1e-6
        y_pos = y.copy()
        z_pos = z.copy()
        y_pos[y_pos < 0] = eps
        z_pos[z_pos < 0] = eps
    elif version == 4: # normalize the values between 0 and 1
        eps = 1e-6
        y_pos = (y - y.min()) / (y.max() - y.min()) + eps
        z_pos = (z - z.min()) / (z.max() - z.min()) + eps
    return 1/2 * (np.sum(y_pos * np.log(y_pos/z_pos)) + np.sum(z_pos * np.log(z_pos/y_pos)))


'''
    This function returns the intelligibility score between Z and Y.
    Arguments:
        Z: posterior features of test speech (list)
        Y: posterior features of reference speech (list)
'''
def match_score(Y, Z, version=1):
    M, N = len(Y), len(Z)
    L = np.zeros((M, N))
    L[0, 0] = KL_diveregence(Y[0], Z[0], version)
    for i in range(1, M):
        L[i, 0] = KL_diveregence(Y[i], Z[0], version) + L[i-1, 0]
    for i in range(1, N):
        L[0, i] = KL_diveregence(Y[0], Z[i], version) + L[0, i-1]
    for i in range(1, M):
        for j in range(1, N):
            L[i, j] = KL_diveregence(Y[i], Z[j], version) + np.min([L[i-1, j], L[i, j-1], L[i-1, j-1]])
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
