import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
import numpy as np

def generate_lc(trace, result, target=2, low=0, high=1, cmap='plasma'):
    x = np.arange(0, len(trace))
    points = np.array([x, trace]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(low, high)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(result[:, target])
    return lc

def pick_traces(traces, results, num_plots=40):
    rand_row = np.random.choice(np.arange(0, traces.shape[0]), num_plots, replace=False)
    
    bool_rows = np.empty(traces.shape[0]).astype(bool)
    bool_rows[:] = False
    bool_rows[rand_row] = True

    sel_traces = traces[bool_rows, :]
    sel_results = results[bool_rows, :, :]

    return sel_traces, sel_results

def display_results(traces, results, row=10, col=4, size=(11.69, 8.27), lw=2, ylim=(0, 4)):
    num_plots = row * col
    sel_traces, sel_results = pick_traces(traces, results, num_plots)

    fig, ax = plt.subplots(row, col, figsize=size, sharey=True, sharex=True)

    for n, a in enumerate(ax.flatten()):
        lc = generate_lc(sel_traces[n, :], sel_results[n, :, :])
        lc.set_linewidth(lw)
        line = a.add_collection(lc)

    xlim = (0, sel_traces.shape[1])
    plt.setp(ax, xlim=xlim, ylim=ylim)
    fig.colorbar(line, ax=ax)
    plt.show()

