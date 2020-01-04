import keras
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Input, Activation

def model_generator(input_dims=(64, 1, 3), steps=3, conv_layers=2, transfer=False, filters=8, kernel_size=8, strides=1, activation='relu', padding='same'):
    '''
    input_dims should be tuple
    steps is number of pooling and upsampling steps
    conv_layers is number of conv1d layers per step
    transfer: if True, will cut and paste encoding layers to decoding layers (as in unet paper)
    '''

    x = Input(shape=(input_dims[0], input_dims[1]))
    y = pooling_module(x, [steps, conv_layers], filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)
 
    base = conv_layer_module(y[-1], conv_layers, filters=filters * (2**steps))

    if transfer:
        for n in y:
            pass
    else:
        filters = base.shape[-1]
        z = upsampling_module(base, [steps, conv_layers], filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)

    output = Conv1D(input_dims[2], input_dims[1])(z[-1])
    output = Activation('softmax')(output)

    return keras.models.Model(x, output)
    
def conv_layer_module(layer, steps, filters=8, kernel_size=8, strides=1, activation='relu', padding='same'):
    '''
    does conv1d followed by batchnormalization. 
    if kernel_size is list, it specifies kernel_Size for each step
    '''

    for s in range(0, steps):

        if kernel_size is list:
            kk = kernel_size[-s]
        else:
            kk = kernel_size

        layer = Conv1D(filters=filters, kernel_size=kk, strides=strides, activation=activation, padding=padding)(layer)
        layer = BatchNormalization()(layer)

    return layer
    
def pooling_module(layer, steps, filters=8, kernel_size=8, strides=1, activation='relu', padding='same'):
    '''
    steps should be list. steps[0] = pooling layers, steps[1] = convolutional steps
    if kernel size is list, it specifies new kernel size for each layer. Can be nested list.
    '''        
    stack = []
    for s in range(0, steps[0]):

        if kernel_size is list:
            kk = kernel_size[-s]
        else:
            kk = kernel_size

        layer = conv_layer_module(layer, steps=steps[1], filters=filters, kernel_size=kk, strides=strides, activation=activation, padding=padding)
        layer = MaxPooling1D()(layer)
        stack.append(layer)

        filters *= 2

    return stack

def upsampling_module(layer, steps, filters=8, kernel_size=8, strides=1, activation='relu', padding='same'):
    '''
    steps should be list. steps[0] = upsampling layers, steps[1] = convolutional steps
    '''
       
    stack = []
    for s in range(0, steps[0]):

        if kernel_size is list:
            kk = kernel_size[-s]
        else:
            kk = kernel_size

        layer = UpSampling1D()(layer)
        layer = conv_layer_module(layer, steps=steps[1], filters=int(filters), kernel_size=kk, strides=strides, activation=activation, padding=padding)
        stack.append(layer)

        filters *= 0.5

    return stack