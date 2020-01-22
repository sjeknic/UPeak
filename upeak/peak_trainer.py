import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os.path import join
from pathlib import Path
from utils.data_processing import label_adjuster

def start_gui(path, output_dir='.', y_limit=[0, 4]):
    global latest_plot, latest_x, traces, plot_count, all_plots, stack_size, save_dir, yaxis

    traces = np.load(path)
    print('Dimensions: {0}'.format(traces.shape))

    save_dir = output_dir
    Path(save_dir).mkdir(parents=False, exist_ok=True)

    yaxis = [float(y) for y in y_limit]
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

def make_next_plot(n):
    global stack_size, traces, yaxis

    plt.clf()
    plt.plot(traces[n])
    plt.title('Plot Number {0}/{1}'.format(n+1, stack_size))
    plt.ylim(yaxis)
    plt.draw()

def click(event):
    global latest_x, latest_y
    latest_x = event.xdata
    latest_y = event.ydata
    
def button(event):
    global latest_x, latest_y, latest_plot, all_plots, plot_count, save_dir
    
    if (event.key == '1') or (event.key == '2'): #saving point
        if latest_x is not None:
            x_spot = int(round(latest_x))
            latest_plot[x_spot] = int(event.key)
            print('Saved point: {0}'.format(x_spot))

            if not np.isnan(traces[plot_count][x_spot]):
                plt.plot(latest_x, traces[plot_count][x_spot], 'o', color=['g', 'r'][int(event.key)-1], markersize=10)
                plt.draw()

            else:
                plt.plot(latest_x, latest_y, 'o', color=['g', 'r'][int(event.key)-1], markersize=10)
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
