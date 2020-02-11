# UPeak
CNN to detect heterogeneous peaks in 1D timeseries data based on the structure of the 2D UNet used for image segmentation.

## Basics

The CNN is currently set up to recognize three output features in-place: background, slope of a peak, plateau of a peak. The output of a model will be an *n x m x 3* matrix, where *n* is the number of traces, *m* is the number of frames, and 3 is for the three output features. 

## Training

Training is done by calling the train function with the raw traces and the generated training labels (see training data below). In the output directory, weights files will be saved at the end of each epoch as well as at the end of training. Additionally, files defining the model structure will also be saved in that directory.

```
python upeak/train.py -t /path/to/traces -l /path/to/labels -o /path/to/output_directory
```

There are many ways to customize the training. The most common is to normalize and/or augment the training data. Adding the *-a* or *-n* tags will apply agumentation or normalization to the data, respectively. Augmentation will always occur before normalization. The functions and parameters for augmentation and normalization are defined in *upeak/_setting.py*. Note that any normalization applied during this stage will also have to be applied when doing predictions.

```
python upeak/train.py -t /path/to/traces -l /path/to/labels -o /path/to/output_directory -a -n
```

Other tags can also be used to adjust the batch size (*-b*), the number of epochs (*-e*), and the number of steps (*-s*). Note that if the number of steps is greater than the available training data, all of the data will be used by the model. Additionally, weights (*-w*) can be defined to prioritize certain output features. For example, to use a batch size of 16, for 8 epochs, with 8000 steps, and a 10x emphasis on finding the plateaus of peaks (output feature 3), the following command can be used.

```
python upeak/train.py -t /path/to/traces -l /path/to/labels -o /path/to/output_directory -a -n -b 16 -e 8 -s 8000 -w 1 1 10
```

Finally, if you are using a custom model structure (see below), the path to the *model_dict.json* file can be provided and the model will use that structure. Note that the input and output dimensions during training are inferred from the structure of the traces and labels provided. The only dimension that needs to be defined in the custom model is the length of trace that it accepts.

```
python upeak/train.py -t /path/to/traces -l /path/to/labels -o /path/to/output_directory -m /path/to/custom/model_dict.json -a -n
```

## Predictions

Predicting traces can be done just by giving the weights file (.hdf5, .h5) and traces (numpy array) to the predict function. The output will be a single numpy array file that is *n x m x c*, where *n* is the number of traces, *m* is the length of each trace, and *c* is the number of output features (default is 3). 

```
python upeak/predict.py -t /path/to/traces -w /path/to/weights -o /path/to/output_directory
```

As before, the *-n* tag can be used to apply normalizations that are defined in *upeak/_setting.py*. Similarly, if a custom model structure was used to during training, the same *model_dict.json* file should be provided here.

```
python upeak/predict.py -t /path/to/traces -w /path/to/weights -o /path/to/output_directory -m /path/to/custom/model_dict.json -n
```

In order to keep the input trace length compatible with the model structure being used, traces will be padded to the appropriate length depending on the model structure. By default, the traces are padded with 0, but this can be changed in *upeak/_setting.py*.

If you want to display the results or save some figures with the predictions overlayed over the trace, you can use the *-d* or *-s* tags, respectively. The settings for this are in *upeak/_setting.py*.

Finally, if you are using a model with a different number of output features, you need to provide that information to the predict function using the *-c* tag. For example, if you have four features, you could use:

```
python upeak/predict.py -t /path/to/traces -w /path/to/weights -o /path/to/output_directory -m /path/to/custom/model_dict.json -n -s -c 4
```

## Generating training data

UPeak has a GUI that can be used to annotate traces. It requires as input a numpy array with the raw trace data. The y-axis limits can be specified using *-y*, and to linearly interpolate nans in the data add the *-n* tag.

```
python upeak/peak_trainer.py -t /path/to/traces -o /path/to/output_dir -y 0 1000 -n
```

In general, the GUI takes inputs in the form of clicking the graph and then pressing a button on the keyboard. Pressing 1, 2, or 3 will mark those points on the graph and save them in the trace. Pressing 1 marks a green point and designates the beginning of a peak. Pressing 2 marks a red point and designates the end of a peak. Pressing 3 marks a cyan point and marks the beginning and end of a plateau. Importantly, there must be two cyan points for every plateau in order for the peak to be appropriately marked. It is fine to designate half peaks (e.g the start of a peak that does not have an end before the final time point). Other than that, it is expected that each peak will have one green point at the start, one red point at the end, and two cyan points designating the plateau. 

The following functions exist in the GUI. To use any of the saving functions, a point on the graph must first be clicked. The terminal window used to launch the GUI will provide feedback for every function it executes so that users can see what is happening.

```
1, 2, or 3 - Mark a point
0 - Delete a point (must click the point to delete)
r - reset the current trace (removes all points and returns y-axis to default)
n - save the current trace and move on to the next trace
b - go back to the previous trace (removes all points from that trace, so it must be re-annotated)
x - save all traces done so far in the output dir (creates traces.npy and labels.npy files)
y - click the top half or the bottom half of the trace, then press y, to increase or decrease the upper limit on the y-axis
c - display the current trace (will apply label_adjuster, so this is what the final output will look like)
```

The peak trainer is automatically set up to use the current labeling scheme (0 - background, 1 - slope, 2 - plateau). Advanced users can modify this behavior to any labeling scheme they want by providing a different function called *label_adjuster* in *upeak/utils/data_processing*. The GUI saves each point marked by the user and the corresponding label, and *label_adjuster* uses these labels to appropriately label the rest of the trace. In this case, the functions used to mark peaks and extract data may not work.

## Custom model structures

## Options in _setting.py