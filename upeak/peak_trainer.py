import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os.path import join
from pathlib import Path
from utils.data_processing import label_adjuster, nan_helper_2d

def start_gui(path, output_dir='.', y_limit=[0, 4]):
    global latest_plot, latest_x, traces, plot_count, all_plots, stack_size, save_dir, yaxis, curr_axis, temp_trace

    traces = np.load(path)
    temp_trace = np.copy(traces)
    temp_trace = nan_helper_2d(temp_trace)
    print('Dimensions: {0}'.format(traces.shape))

    save_dir = output_dir
    Path(save_dir).mkdir(parents=False, exist_ok=True)

    yaxis = [float(y) for y in y_limit]
    curr_axis = yaxis
    plot_count = 0
    stack_size = traces.shape[0]
    latest_x = None
    latest_plot = np.zeros(traces.shape[1])
    all_plots = None

    fig = plt.figure()
    plt.plot(traces[plot_count])
    plt.title('Plot Number {0}/{1}'.format(plot_count+1, stack_size))
    plt.ylim(yaxis)

    cid_1 = fig.canvas.mpl_connect('button_press_event', click)
    cid_2 = fig.canvas.mpl_connect('key_press_event', button)

    plt.show()

def make_next_plot(n, temp_yaxis=None):
    global stack_size, traces, yaxis, curr_axis

    plt.clf()
    plt.plot(traces[n])
    plt.title('Plot Number {0}/{1}'.format(n+1, stack_size))
    if temp_yaxis is None:
        plt.ylim(yaxis)
        curr_axis = yaxis
    else:
        plt.ylim(temp_yaxis)
        curr_axis = temp_yaxis
        redraw_points()
    plt.draw()

def redraw_points():
    global latest_plot, plot_count, traces, temp_trace

    idxs = np.where(latest_plot>0)[0]
    for i in idxs:
        if not np.isnan(traces[plot_count][i]):
            plt.plot(i, traces[plot_count][i], 'o', color=['g', 'r', 'c'][int(latest_plot[i] - 1)], markersize=8)
        else:
            plt.plot(i, temp_trace[plot_count][i], 'o', color=['g', 'r', 'c'][int(latest_plot[i] - 1)], markersize=8)

def click(event):
    global latest_x, latest_y
    latest_x = event.xdata
    latest_y = event.ydata
    
def button(event):
    global latest_x, latest_y, latest_plot, all_plots, plot_count, save_dir, yaxis, curr_axis, traces
    
    if (event.key == '1') or (event.key == '2') or (event.key == '3'): #saving point
        if latest_x is not None:
            x_spot = int(round(latest_x))
            latest_plot[x_spot] = int(event.key)
            print('Saved point: {0}'.format(x_spot))

            if not np.isnan(traces[plot_count][x_spot]):
                plt.plot(x_spot, traces[plot_count][x_spot], 'o', color=['g', 'r', 'c'][int(event.key)-1], markersize=8)
                plt.draw()

            else:
                plt.plot(latest_x, latest_y, 'o', color=['g', 'r', 'c'][int(event.key)-1], markersize=8)
                plt.draw()
            
            latest_x = None
            latest_y = None 

        else:
            print('No point saved.')

    elif event.key == 'r': #reset current trace
        latest_plot = np.zeros(latest_plot.shape)
        print('Trace reset.')
        make_next_plot(plot_count)

    elif event.key == 'b': #go back to previous trace
        if not plot_count <= 0:
            all_plots = all_plots[:-1, :]
            plot_count += -1
            print('Deleted last trace. New size: {0}'.format(all_plots.shape))
            make_next_plot(plot_count)
        else:
            print('No trace to delete.')

    elif event.key == 'x': #save all traces up until now. Must be done after 'n' to include current trace.

        np.save(join(save_dir, 'trained.npy'), all_plots)
        np.save(join(save_dir, 'traces.npy'), traces[:plot_count, :])  
        print('Saved {0} traces in {1}'.format(plot_count+1, save_dir))

    elif event.key == 'n': #save current trace and move to the next one
        latest_plot = label_adjuster(latest_plot)

        if all_plots is None:
            all_plots = latest_plot
            latest_plot = np.zeros(latest_plot.shape)
        else:
            all_plots = np.vstack((all_plots, latest_plot))
            latest_plot = np.zeros(latest_plot.shape)
            
        print('Trace {0} saved.'.format(plot_count))
        
        try:
            plot_count += 1
            make_next_plot(plot_count)
        except IndexError:
            print('End of stack.')
            plt.clf()
            plt.draw()

    elif event.key == 'y':
        '''
        this will be for adjusting the yaxis
        need to either register mutliple inputs or come up with a clever way of distinguishing zoom in v zoom out
        '''
        if latest_y is not None:
            if int(round(latest_y)) >= np.max(curr_axis) / 2:
                if np.max(curr_axis) <= 10:
                    new_ymax = np.max(curr_axis) + 1
                elif np.max(curr_axis) <= 100:
                    new_ymax = np.max(curr_axis) + 10
                else:
                    new_ymax = np.max(curr_axis) + 100
            elif int(round(latest_y) < np.max(curr_axis) / 2):
                if np.max(curr_axis) <= 2:
                    new_ymax = np.max(curr_axis) - 0.5
                elif np.max(curr_axis) <= 10:
                    new_ymax = np.max(curr_axis) - 1
                elif np.max(curr_axis) <= 100:
                    new_ymax = np.max(curr_axis) - 10
                else:
                    new_ymax = np.max(curr_axis) - 100

            print('Rescaled y-axis to {0}'.format([np.min(curr_axis), new_ymax]))
            make_next_plot(plot_count, temp_yaxis=[np.min(curr_axis), new_ymax])
        else:
            print('No click data available to scale axis. Click plot first, then press y.')
        pass

    elif event.key == 'c': #check current trace
        print(latest_plot)

    else:
        print('No function for key {0}'.format(event.key))

def _parse_args():
    parser = argparse.ArgumentParser(description='peak trainer')
    parser.add_argument('-t', '--traces', help='path to .npy file with raw traces')
    parser.add_argument('-o', '--output', help='output directory', default='.')
    parser.add_argument('-y', '--yaxis', help='bounds for y axis', default=[0, 4], nargs='*')
    return parser.parse_args()

def _main():
    args = _parse_args()

    start_gui(args.traces, args.output, args.yaxis)

if __name__ == '__main__':
    _main()
