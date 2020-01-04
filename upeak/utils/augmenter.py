import numpy as np

def noise(arr, loc=1, scale=0.05):
    return arr * np.random.normal(loc=loc, scale=scale, size=arr.shape)

def amplitude(arr, scale=1000):
    return arr * scale * np.random.rand(arr.shape[0])[:, np.newaxis]

def augment(arr):
    '''
    Should return original array stacked on an array that has noise added and amplitude adjusted.
    '''
    narr = noise(arr)
    narr = amplitude(narr)
    return np.vstack([arr, narr])