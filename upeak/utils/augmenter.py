import numpy as np

def noise(arr, loc=1, scale=0.05):
    return arr * np.random.normal(loc=loc, scale=scale, size=arr.shape)

def amplitude(arr, scale=1000):
    return arr * scale * np.random.rand(arr.shape[0])[:, np.newaxis]

def augment_data(arr):
    '''
    Should return original array stacked on an array that has noise added and amplitude adjusted.
    '''
    narr = noise(arr)
    narr = amplitude(narr)
    return np.vstack([arr, narr])

def gen_augment_arr(dims, frac=0.5):
    '''
    Returns arr of size dims, with frac rows containing augmented data
    Should include option in the future to select augmenting fucntions and parameters
    '''

    arr = np.ones(dims)
    narr = amplitude(noise(arr))
    narr[::int(1/frac)] = 1
    return narr
