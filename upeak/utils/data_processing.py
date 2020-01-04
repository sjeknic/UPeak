import numpy as np
import argparse
from keras.utils import to_categorical, Sequence
from utils.augmenter import gen_augment_arr

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

def label_adjuster(labels):
    # messy, should be done on whole array at once, not one row at a time
    # also, should be done in place

    lab_strat = None
    for l in labels:
        ones = np.where(l==1)[0]
        twos = np.where(l==2)[0]

        for n, o in enumerate(ones):
            o = int(o)
            try:
                if twos[n] > o:
                    l[o+1:twos[n]] = 2
                    l[twos[n]] = 1
            except:
                l[o+1] = 2

        if lab_strat is None:
            lab_strat = l
        else:
            lab_strat = np.vstack([lab_strat, l])

    return lab_strat

def stack_sequences(seq_list, cv=np.nan):
    '''
    Input list of 2d arrays. pads ends of traces with nan to same length and stacks traces
    Still needs to be tested for several different sizes of arrays.
    '''
    l_max = np.max([a.shape[1]for a in seq_list])
    seq_list = [np.pad(a, ((0, 0), (0, l_max - a.shape[1])), constant_values=cv) for a in seq_list]
    return np.vstack(seq_list)

def load_data(traces, labels):
    '''
    Loading will probably fail if traces are of different lengths
    '''
    traces = stack_sequences([nan_helper_2d(np.load(t)) for t in traces], cv=np.nan)
    labels = stack_sequences([np.load(l) for l in labels], cv=0)

    traces = np.expand_dims(traces, axis=-1)

    labels = label_adjuster(labels) # this should probably be done before loading in this function
    labels = to_categorical(labels)

    return traces, labels

def pick_positions(traces, labels, num, length=64):
    '''
    inputs array of traces and labels. returns num traces and labels of length len.
    would be better to check for nans before the position is picked, as opposed to after, but I'll see how it works
    '''
    first_nan = np.logical_not(np.isnan(traces)).argmin(axis=1) #loc of first nan in each row
    
    rand_row = np.random.randint(0, traces.shape[0], num)
    rand_col = np.random.randint(0, traces.shape[1] - length, num)
    
    #probably a faster way to do this...
    for n, r in enumerate(rand_row):
        c = rand_col[n]
        if ((c + length) >= first_nan[r]) and (first_nan[r]>0):
            rand_col[n] = np.random.randint(0, first_nan[r]-length)
            
    return list(zip(rand_row, rand_col))

class DataGenerator(Sequence):
    def __init__(self, traces, labels, length=64, batch_size=32, steps=1000, shuffle=True, augment=False):
        self.traces = traces
        self.labels = labels
        self.length = length
        self.batch_size = batch_size
        self.steps = steps
        self.list_idxs = pick_positions(traces, labels, self.batch_size*self.steps)
        self.idxs = np.arange(len(self.list_idxs))
        self.shuffle = shuffle
        self.augment = augment
    
    def __len__(self):
        'num batches per epoch'
        return int(np.floor(len(self.list_idxs))/self.batch_size)
    
    def __getitem__(self, index):
        'returns one batch of data'
        #indexs for the batch
        indexs = self.idxs[index*self.batch_size:(index+1)*self.batch_size]

        #list of points
        data_idxs = [self.list_idxs[i] for i in indexs]
        
        x, y = self.__data_generation(data_idxs)

        if self.augment == True:
            #sloppy
            aug_arr = gen_augment_arr((x.shape[0], x.shape[1]))
            x = x[:,:,0] * aug_arr
            x = np.expand_dims(x, axis=-1)
        
        return x, y
    
    def on_epoch_end(self):
        'randomize order after each epoch'
        self.idxs = np.arange(len(self.list_idxs))
        if self.shuffle == True:
            np.random.shuffle(self.idxs)
    
    def __data_generation(self, data_idxs):
        t_patch = np.array([self.traces[r, c:c+self.length] for (r, c) in data_idxs])
        l_patch = np.array([self.labels[r, c:c+self.length, :] for (r, c) in data_idxs])
        
        return t_patch, l_patch

def _parse_args():

    parser = argparse.ArgumentParser(description='data processor')
    parser.add_argument('-n', '--nans', help='add if nans need to be removed', default=1)
    parser.add_argument('-l', '--labels', help='add if label categories need to be set', default=1)
    parser.add_argument('-o', '--output', help='where to save the output array', default='.')
    parser.add_argument('-i', '--input', help='path to input array')
    return parser.parse_args()

def _main():

    args = _parse_args()
    arr = np.load(args.input)

    if not args.nans: #runs nan cleaning if args.nans is false
        arr = nan_helper_2d(arr)
        np.save(args.output, arr)

    if not args.labels: #runs label cleaning if args.labels is false
        arr = label_adjuster(arr)
        np.save(args.output, arr)

if __name__ == '__main__':
    _main()