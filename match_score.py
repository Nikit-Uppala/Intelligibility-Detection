import numpy as np

def KL_diveregence(y, z, version):
    """This function calculates the KL divergence, according to Equation-2, in the paper. 
    Utterance Verification-Based Dysarthric Speech Intelligibility Assessment Using Phonetic Posterior Features

    Args:
        y (nparray): posterior features of reference speech (list)
        z (nparray): posterior features of test speech (list)
        version (int): method of processing -ve values, for more details se the comments in the function

    Returns:
        float: local match score computed as symmetric Kullback-Leibler divergence between y and z
    """
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
    # return 1/2 * (np.sum(y_pos * np.log(y_pos/z_pos)) + np.sum(z_pos * np.log(z_pos/y_pos)))
    # return np.mean(np.abs(y_pos - z_pos))
    return np.square(np.subtract(y_pos, z_pos)).mean()



def match_score(Y, Z, version=1):
    """This function returns the accumulated match score (inteligibility score) between Z and Y.

    Args:
        Y (list): posterior features of reference speech
        Z (list): posterior features of test speech
        version (int, optional): _description_. Defaults to 1.

    Returns:
        float: accumulated match score (inteligibility score)
    """
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
