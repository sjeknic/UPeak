# TEST SET SETTINGS

FRAC_TEST = 0.2  # fraction of input data to be used as validation data
VAL_STEPS = 10000 # max number of traces that will be pulled from the validation set for validation

# AUGMENTATION SETTINGS

AUG_FUNCS = ['noise'] #options are 'noise', 'amplitude', and 'filter'. Will be run in order of list
AUG_OPTIONS = [{'loc': 1, 'scale': 0.10}] #kwargs for the augmentation functions that are being used. if no options are provided, just uses default
AUG_METHOD = ['stack'] #options are stack (which adds the traces to the end of the training stack) or inplace (which does augmentation on the training arr)

# NORMALIZATION SETTINGS

NORM_FUNCS = ['zscore'] #options are zscore, amplitude, gradient, maxabs, norm. Including both should only work with concatenation method
NORM_OPTIONS = [{'normalize':True}] #kwargs for the normalization functions above
NORM_METHOD = ['inplace'] #options are inplace or concatenate. If concatenate, the model must be set up to take input vectors with greater than one feature.

# FILE NAMES
PRED_FNAME = 'predictions'
WEIGHT_FNAME = 'model_weights'

# ACTIVATION SETTINGS 

ALPHA = 0.3 # alpha for paramaterized relu or LeakyReLU

# PAD SETTINGS

PAD_MODE = 'constant'
PAD_CV = 0

# DISPLAY/SAVE SETTINGS

DISP_ROWS = 8
DISP_COLS = 4
DISP_SIZE = (11.69, 8.27) #inches
DISP_YLIM = [0, 6]
DISP_LW = 2
DISP_NUMFIGS = 20
DISP_CLASS = -1 
DISP_CMAP = 'plasma'
DISP_FORMAT = 'png'