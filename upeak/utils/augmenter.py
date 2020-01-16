import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def augment_decorator(func):
    def wrapper(arr, method, **kwargs):
        new_arr = func(arr, **kwargs)

        if method == 'stack':
            return np.vstack([arr, new_arr])
        elif method == 'concatenate':
            return np.concatenate([arr, new_arr], axis=-1)
        elif method == 'inplace':
            return new_arr
        else:
            raise ValueError('Unknown method: {0}'.format(method))
    return wrapper

@augment_decorator
def noise(arr, loc=1, scale=0.05):
    return arr * np.random.normal(loc=loc, scale=scale, size=arr.shape)

@augment_decorator
def amplitude(arr, scale=1000):
    return arr * scale * np.random.rand(arr.shape[0])[:, np.newaxis, np.newaxis]

@augment_decorator
def no_change(arr):
    return arr

def augment_data(arr):
    '''
    NO LONGER USED
    Should return original array stacked on an array that has noise added and amplitude adjusted.
    '''
    narr = noise(arr)
    narr = amplitude(narr)
    return np.vstack([arr, narr])

def _augment(funcs, options, method, traces, labels):
    '''
    '''
    assert len(funcs) == len(method), 'Functions and methods must be same length'
    func_dict = {'noise' : noise, 'amplitude' : amplitude}
    to_run = [func_dict[f] for f in funcs]
    
    while len(to_run) > len(options):
        options.append({})

    aug_traces = traces.copy()
    aug_labels = labels.copy()

    for t, m, o in zip(to_run, method, options):
        aug_traces = t(aug_traces, m, **o)
        aug_labels = no_change(aug_labels, m)

    return aug_traces, aug_labels

def _normalize(funcs, options, method, arr):
    '''
    '''
    assert len(funcs) == len(method), 'Functions and methods must be same length'
    func_dict = {'zscore' : normalize_zscore, 'amplitude' : normalize_amplitude}
    to_run = [func_dict[f] for f in funcs]
    
    while len(to_run) > len(options):
        options.append({})
 
    normed = arr.copy()

    for t, m, o in zip(to_run, method, options):
        normed = t(normed, m, **o)

    return normed

def gen_augment_arr(dims, frac=0.5):
    '''
    NO LONGER USED
    Returns arr of size dims, with frac rows containing augmented data
    Should include option in the future to select augmenting fucntions and parameters
    '''
    arr = np.ones(dims)
    #narr = amplitude(noise(arr))
    narr = noise(arr)
    narr[::int(1/frac)] = 1
    return narr

def filter_nonresponders(traces, labels, frac=0.5, thres=0.02, filter=2):
    '''
    frac is fraction of traces with no peaks to remove.
    thres is percent of cells that have to be positive in order to be considered a responding trace
    filter is the classification used for filtering
    labels should be size of (traces, traces, filter)
    '''
    thres *= traces.shape[1]
    positive_points = np.array([labels[n, :, filter].sum() for n in range(0, labels.shape[0])])
    non_responders = positive_points <= thres

    traces_responders = traces[~non_responders, :, :].copy()
    labels_responders = labels[~non_responders, :, :].copy()

    traces_nonresponders = traces[non_responders, :, :].copy()
    labels_nonresponders = labels[non_responders, :, :].copy()

    traces_nonresponders = traces_nonresponders[::int(1/frac)]
    labels_nonresponders = labels_nonresponders[::int(1/frac)]

    return np.vstack([traces_responders, traces_nonresponders]), np.vstack([labels_responders, labels_nonresponders])

@augment_decorator
def normalize_zscore(traces, by_row=True, offset=0):
    '''
    by_row, if true will normalize each trace individually. False will normalize the whole stack together
    offset can be added to prevent negative values
    '''
    def z_func(a, offset=offset):
        return stats.zscore(a, nan_policy='omit') + offset

    if by_row:
        return np.apply_along_axis(z_func, 1, traces) # (function, axis, array)
    else:
        return z_func(traces)

@augment_decorator
def normalize_amplitude(traces, by_row=True):
    '''
    method should be either amplitude or zscore
    by_row, if true will normalize each trace individually. False will normalize the whole stack together
    '''
    if by_row:
        row_maxs = np.nanmax(traces, axis=1)
        return traces / row_maxs[:, np.newaxis]
    else:
        return traces / np.nanmax(traces)