import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
import numpy as np
from _setting import DISP_ROWS, DISP_COLS, DISP_SIZE, DISP_YLIM, DISP_LW, DISP_NUMFIGS, DISP_CLASS, DISP_CMAP, DISP_FORMAT
from os.path import join
from pathlib import Path

print(DISP_NUMFIGS)

def generate_lc(trace, result, target=DISP_CLASS, low=0, high=1, cmap=DISP_CMAP):
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

def display_results(traces, results, row=DISP_ROWS, col=DISP_COLS, size=DISP_SIZE, lw=DISP_LW, ylim=DISP_YLIM):
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

def save_figures(traces, results, save_path, numfig=DISP_NUMFIGS, row=DISP_ROWS, col=DISP_COLS, size=DISP_SIZE, lw=DISP_LW, ylim=DISP_YLIM):
    
    Path(save_path).mkdir(parents=False, exist_ok=True)

    num_plots = row * col

    if numfig * row * col >= traces.shape[0]:
        #trying to make more plots than traces
        numfig = np.floor(traces.shape[0] / (row * col))

    for i in range(numfig):
        fig, ax = plt.subplots(row, col, figsize=size, sharey=True, sharex=True)

        for n, a in enumerate(ax.flatten()):
            lc = generate_lc(traces[n + (i * num_plots), :], results[n + (i * num_plots), :, :])
            lc.set_linewidth(lw)
            line = a.add_collection(lc)

        xlim = (0, traces.shape[1])
        plt.setp(ax, xlim=xlim, ylim=ylim)
        fig.colorbar(line, ax=ax)

        plt.savefig(join(save_path, 'fig_{0}.{1}'.format(i, DISP_FORMAT)), format=DISP_FORMAT)
        plt.close()
