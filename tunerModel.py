import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, BatchNormalization, Dropout, AveragePooling1D, SpatialDropout1D
from tensorflow.keras import Model
import keras_tuner as kt

def build_model(hp):
    dropout = hp.Float('dropout', .02, .2, step=.02)
    sDropout = hp.Float('sparseDropout', 0, .14, step=.02)
    activation = 'elu'
    #activation = hp.Choice('activaion', values=['elu', 'sigmoid', 'selu'])
    #convActivation = hp.Choice('convActivation', values=['elu', 'sigmoid', 'exponential', 'selu'])

    #l2 = hp.Choice('l2', values=[1e-4, 1e-5, 1e-6, 0.0])
    l2 = hp.Float('l2', 1e-8, 1e-4, sampling = 'log')

    regularizer = None
    if l2 > 0:
        regularizer = keras.regularizers.l2(l2)

    #I assume 4 dense layers
    dense1 = hp.Int('dense1', 12, 90, step=6)
    dense2 = hp.Int('dense2', 12, 90, step=6)
    dense3 = hp.Int('dense3', 12, 72, step=6)
    dense4 = hp.Int('dense4', 12, 36, step=6)
    dense_layers = [dense1, dense2, dense3, dense4]

    learningRate = hp.Choice('learningRate', values=[1e-2, 1e-3, 1e-4], ordered = True)
    
    timeSteps = 1500
    reduction = hp.Int('reduction', 1, 4, step = 1, default = 1) 
    timeSteps = timeSteps / reduction

    kernelSizes = []
    poolSizes = []
    poolStrides = []
    filters = []

    #give it up 16 layers
    for layer in range(1,17):
        k = hp.Int('kernelSize' + str(layer), 1, 11, step = 2, default = 7)
        pSize = hp.Int('poolSize' + str(layer), 3,7, step = 1, default = 3)
        pStride = hp.Int('poolStride' + str(layer), 1,3, step = 1, default = 1)#, sampling = 'log')
        f = hp.Int('filter' + str(layer), 1, 71, step = 5, default = 50)

        if (timeSteps - (k - 1)) / pStride >= 1:
            timeSteps = int((timeSteps - (k - 1)) / pStride)
            kernelSizes.append(k)
            poolSizes.append(pSize)
            poolStrides.append(pStride)
            filters.append(f)

    conv_layers = [{'filters': f, 'kernelSize': k, 'poolSize': p, 'poolStrides': ps}
            for f, k, p, ps in zip(filters, kernelSizes, poolSizes, poolStrides)]

    #I'm just going to hardcode these because they probably aren't changing
    nclasses=5
    timeSteps = 1500
    channels = 51

    model = create_model(timeSteps, channels, nclasses, conv_layers, reduction, dense_layers, dropout, sDropout, regularizer, activation, learningRate)
    return model

def create_model(ntimeSteps, nchannels, nclasses, convLayers, reduction, denseLayers, dropout, sDropout, regularizer, activation, learningRate):
    inputTensor = Input(shape = (ntimeSteps, nchannels), name = 'input')

    previousTensor = inputTensor #previousTensor is used to link all of the tensors together

    #Used to reduce the timesteps immediately
    previousTensor=AveragePooling1D(
                pool_size=reduction, 
                strides = reduction,
                name='AdvPool',
                padding = 'same'
            )(previousTensor)
    
    
    count=0 #used to name layers
    #Creates the convolution layers
    for i in convLayers:
        name = 'C%02d'%count
        previousTensor = Convolution1D(
            filters = i['filters'],
            kernel_size = i['kernelSize'],
            strides = 1,
            padding = 'valid',
            use_bias = True,
            kernel_initializer = 'random_uniform',
            bias_initializer = 'zeros',
            name = name,
            activation = activation,
            kernel_regularizer=regularizer
        )(previousTensor)

        if(i['poolStrides'] > 1):
            name='Max%02d'%count
            previousTensor=MaxPooling1D(
                pool_size=i['poolSize'], 
                strides = i['poolStrides'],
                padding = 'same',
                name=name
            )(previousTensor)
        
        if sDropout > 0:
            name='SpatialDrop%.02d'%count
            previousTensor=SpatialDropout1D(sDropout, name=name)(previousTensor)
            
        count+=1

    previousTensor = Flatten()(previousTensor)

    count = 0#used to name layers
    #Creates the dense layers
    for i in denseLayers:
        name = 'D%02d'%count
        previousTensor = Dense(
            units = i,
            activation = activation,
            use_bias = 'True',
            bias_initializer = 'zeros',
            name = name,
            kernel_regularizer=regularizer
        )(previousTensor)

        if dropout > 0:
            name='Drop%02d'%count
            previousTensor=Dropout(dropout, name=name)(previousTensor)
        count+=1

    output = Dense(
        units = nclasses,
        activation = 'softmax',
        bias_initializer = 'zeros',
        name = 'output'
    )(previousTensor)

    model = Model(inputs = inputTensor, outputs = output) #Actually creates the model
    opt = tf.keras.optimizers.Adam(learning_rate=learningRate, beta_1=.9, beta_2=.999, epsilon=None, decay=0.0, amsgrad=False) #Creates the optimizer
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['categorical_accuracy']) #Compiles the model with the optimizer and metrics
    print(model.summary())
    return model
