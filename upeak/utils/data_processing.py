import numpy as np
import argparse

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def nan_helper_2d(arr):
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

def _parse_args():

    parser = argparse.ArgumentParser(description='model trainer')
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

if __name__ == '__main__'
    _main()