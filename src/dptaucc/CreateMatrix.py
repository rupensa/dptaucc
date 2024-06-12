import numpy as np
from itertools import product, islice, filterfalse
from warnings import warn

def CreateMatrix(nrows = None, ncols = None, rowclust = None, colclust = None, noise = 0, v_rowclust = None, v_colclust = None, max_attempts = 1, random_state=None):
    rng = np.random.default_rng(seed = random_state)

    if (rowclust == None or nrows == None) and v_rowclust == None:
        raise ValueError('Error')

    if v_rowclust == None:
        v_rowclust = f(nrows, rowclust)
    else:
        rowclust = len(v_rowclust)
        nrows = sum(v_rowclust)

    if (colclust == None or ncols == None) and v_colclust == None:
        raise ValueError('Error')

    if v_colclust == None:
        v_colclust = f(ncols, colclust)
    else:
        colclust = len(v_colclust)
        ncols = sum(v_colclust)
    
    if (rowclust > nrows) or (colclust > ncols):
        raise ValueError('Error')

    if max(rowclust, colclust) >= 2 ** min(rowclust, colclust):
        raise ValueError('Error')

    m = min(rowclust, colclust)
    M = max(rowclust, colclust)
    
    
    d = list(np.diag([1] * m))
    l1 = [tuple(t) for t in d]
    l2 = list(islice(filterfalse(lambda x: sum(x) == 1, product([0,1], repeat = m)), 1, M-m + 1))
    l = l1 + l2
    l.sort(key = sum)                

    if colclust > rowclust:
        l = np.array(l).T
    else:
        l = np.array(l)
    l = np.append(l, np.arange(rowclust).reshape(rowclust, 1), axis = 1)
    l = np.append(l, np.arange(colclust + 1).reshape(1, colclust + 1), axis = 0)
    V = np.repeat(l, v_colclust + [1], axis = 1)
    V = np.repeat(V, v_rowclust + [1], axis = 0)

    target_r = V[:nrows, -1]
    target_c = V[-1, :ncols]
    V = V[:nrows, :ncols]
    
    V = replaceRandom(V, noise, rng, max_attempts = max_attempts)

    return V, target_r, target_c

def f(n, nclust):
    rows_per_clust = [round(n / nclust), int(n / nclust)]
    a = rows_per_clust * int((nclust + 1) / 2)
    s = sum(a[:nclust - 1])
    a[nclust - 1] = n - s
    v = a[:nclust]
    return v

def replaceRandomBase(arr, noise, rng):
    temp = np.asarray(arr)   # Cast to numpy array
    shape = temp.shape       # Store original shape
    temp = temp.flatten()    # Flatten to 1D
    inds = rng.choice(temp.size, size = round(temp.size * noise ), replace = False)   # Get random indices
    temp[inds] = (temp[inds] + 1) % 2        # Fill with something
    temp = temp.reshape(shape)                     # Restore original shape
    if len(shape) == 3:
        # check for 0-slices
        s01 = np.sum(temp, axis = (0,1))
        s02 = np.sum(temp, axis = (0,2))
        s12 = np.sum(temp, axis = (1,2))
        check = np.sum(s01 == 0) + np.sum(s02 == 0) + np.sum(s12 == 0)
    elif len(shape) == 2:
        s0 = np.sum(temp, axis = 0)
        s1 = np.sum(temp, axis = 1)
        check = np.sum(s0 ==0) + np.sum(s1 == 0)
    else:
        check = 0
    return temp, check

def replaceRandom(arr, noise, rng, max_attempts = 1):
    for i in range(max_attempts):
        temp, check = replaceRandomBase(arr, noise, rng)
        if check == 0:
            break
    if check > 0:        
        warn ('Zero slice', UserWarning)
    return temp

