import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def noise(arr, loc=1, scale=0.05):
    return arr * np.random.normal(loc=loc, scale=scale, size=arr.shape)

def amplitude(arr, scale=1000):
    return arr * scale * np.random.rand(arr.shape[0])[:, np.newaxis, np.newaxis]

def augment_data(arr):
    '''
    Should return original array stacked on an array that has noise added and amplitude adjusted.
    '''
    narr = noise(arr)
    narr = amplitude(narr)
    return np.vstack([arr, narr])

def _augment(funcs, options, method, traces, labels):
    func_dict = {'noise' : noise, 'amplitude' : amplitude}
    to_run = [func_dict[f] for f in funcs]
    
    while len(to_run) > len(options):
        options.append({})

    augmented = traces.copy()

    for t, o in zip(to_run, options):
        augmented = t(augmented, **o)

    if method == 'concatenate':
        return np.vstack([traces, augmented]), np.vstack([labels, labels])
    elif method == 'inplace':
        return augmented, labels
    else:
        raise ValueError('unknown augment method: {0}'.format(method))

def _normalize(funcs, options, method, arr):
    func_dict = {'zscore' : normalize_zscore, 'amplitude' : normalize_amplitude}
    to_run = [func_dict[f] for f in funcs]
    
    while len(to_run) > len(options):
        options.append({})
 
    normed = arr.copy()

    for t, o in zip(to_run, options):
        normed = t(normed, **o)

    if method == 'concatenate':
        return np.concatenate([arr, normed])
    elif method == 'inplace':
        return normed
    else:
        raise ValueError('unknown augment method: {0}'.format(method))

def gen_augment_arr(dims, frac=0.5):
    '''
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