import numpy as np
import argparse
from tensorflow.keras.utils import to_categorical, Sequence
from utils.augmenter import normalize_zscore
from os.path import join
from pathlib import Path

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def nan_helper_2d(arr):
    #probably can be done faster
    temp = np.empty(arr.shape)
    temp[:] = np.nan
    for n, y in enumerate(arr):
        nans, z = nan_helper(y)
        y[nans] = np.interp(z(nans), z(~nans), y[~nans])
        temp[n, :] = y 
    return temp

def label_adjuster(l):
    if not np.count_nonzero(l) == 0:
        ones = np.where(l==1)[0]
        twos = np.where(l==2)[0]
        threes = np.where(l==3)[0]

        if len(ones) > len(twos):
            l[ones[-1]+1:] = 1
            l[ones[-1]] = 0
            ones = ones[:-1]
        elif len(ones) < len(twos):
            l[:twos[0]] = 1
            l[twos[0]] = 0
            twos = twos[1:]

        for o, t in zip(ones, twos):
            if o < t:
                l[o+1:t] = 1
                l[t] = 0
            elif t < o: # this should be case of end of peak at start of trace, start of peak at end of trace
                l[:t] = 1
                l[t] = 0
                l[o+1:] = 1

        for t in range(len(threes))[::2]:
            l[threes[t]:threes[t+1] + 1] = 2

    return l

def pad_traces(traces, model_size, pad_mode='edge', cv=0):
    '''
    pad_mode is edge or constant.
    if edge, repeats last value from trace. if constant, pads with cv
    traces are padded at the end. might be good to add functionality to pad at the start of trace
    '''
    options_dict = {'constant_values':cv} if pad_mode == 'constant' else {}
    target_mod = 2 ** model_size
    diff = target_mod - (traces.shape[1] % target_mod)

    if diff == target_mod:
        # no padding needed
        return traces
    else:
        return np.pad(traces, pad_width=((0, 0), (0, diff), (0, 0)), mode=pad_mode, **options_dict)

def stack_sequences(seq_list, cv=np.nan):
    '''
    Input list of 2d arrays. pads ends of traces with nan to same length and stacks traces
    '''
    l_max = np.max([a.shape[1]for a in seq_list])
    seq_list = [np.pad(a, ((0, 0), (0, l_max - a.shape[1])), constant_values=cv) for a in seq_list]
    return np.vstack(seq_list)

def load_data(traces, labels=None):
    '''
    Loads data. Makes traces correct dimension (3D) and converts labels to categorical
    traces of different lengths are padded.
    '''
    traces = stack_sequences([nan_helper_2d(np.load(t)) for t in traces], cv=np.nan)
    traces = np.expand_dims(traces, axis=-1)

    if labels is not None:
        labels = stack_sequences([np.load(l) for l in labels], cv=0)
        labels = to_categorical(labels)

    return traces, labels

def pick_all_positions(traces, length=128):
    first_nan = np.logical_not(np.isnan(traces)).argmin(axis=1)

    positions = []
    for i, fn in enumerate(first_nan):
        if len(fn) > 1: #required if more than one input feature. All fn should be identical
            fn = fn[0]
        end = fn if fn > 0 else traces.shape[1]
        if end - length >= 0:
            cols = np.arange(0, end - length)   
            positions.extend([(i, y) for y in cols])

    return positions

def pick_random_positions(traces, num, length=64):
    '''
    No longer being used...
    inputs array of traces and labels. positions of traces and labels that will include no nans
    would be better to check for nans before the position is picked, as opposed to after, but I'll see how it works
    '''
    first_nan = np.logical_not(np.isnan(traces)).argmin(axis=1) #loc of first nan in each row
    
    rand_row = np.random.randint(0, traces.shape[0], num)
    rand_col = np.random.randint(0, traces.shape[1] - length, num)
    
    #probably a faster way to do this...
    for n, r in enumerate(rand_row):
        c = rand_col[n]
        if ((c + length) >= first_nan[r, 0]) and (first_nan[r, 0]>0): #remove the second indice in indexing first_nan if you get errors
            rand_col[n] = np.random.randint(0, first_nan[r, 0]-length)
            
    return list(zip(rand_row, rand_col))

def gen_train_test(traces, labels, frac=0.2):
    num_test_rows = int(np.floor(frac * traces.shape[0]))
    test_rows = np.random.choice(traces.shape[0], num_test_rows, replace=False)

    bool_rows = np.empty(traces.shape[0]).astype(bool)
    bool_rows[:] = False
    bool_rows[test_rows] = True

    test_traces = traces[bool_rows, :, :].copy()
    train_traces = traces[~bool_rows, :, :].copy()

    test_labels = labels[bool_rows, :, :].copy()
    train_labels = labels[~bool_rows, :, :].copy()

    return train_traces, train_labels, test_traces, test_labels

class DataGenerator(Sequence):
    def __init__(self, traces, labels, length=64, batch_size=32, steps=500, shuffle=True):
        self.traces = traces
        self.labels = labels
        self.length = length
        self.batch_size = batch_size
        self.steps = steps
        self.shuffle = shuffle
        self.list_idxs = pick_all_positions(traces, length=self.length)

        if len(self.list_idxs) > (self.batch_size * self.steps):
            self.idxs = np.random.choice(len(self.list_idxs), self.batch_size*self.steps, replace=False)
        else:
            self.idxs = np.arange(len(self.list_idxs))

        np.random.shuffle(self.idxs)

    def __len__(self):
        '''num batches per epoch'''
        return int(np.floor(len(self.idxs))/self.batch_size)
    
    def __getitem__(self, index):
        '''returns one batch of data'''
        #indexs for the batch
        indexs = self.idxs[index*self.batch_size : (index+1)*self.batch_size]

        #list of points
        data_idxs = [self.list_idxs[i] for i in indexs]
        
        #extract data
        x, y = self.__data_generation(data_idxs)
        
        return x, y
    
    def on_epoch_end(self):
        '''randomize order after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.idxs)
    
    def __data_generation(self, data_idxs):
        t_patch = np.array([self.traces[r, c:c+self.length, :] for (r, c) in data_idxs])
        l_patch = np.array([self.labels[r, c:c+self.length, :] for (r, c) in data_idxs])
        
        return t_patch, l_patch