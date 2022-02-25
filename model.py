import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, BatchNormalization, Dropout, AveragePooling1D
from tensorflow.keras import Model

def create_model(ntimeSteps, nchannels, nclasses, convLayers, denseLayers, pDropout=0, l2=0):
    regularizer = None
    if l2 > 0:
        regularizer = keras.regularizers.l2(l2)
    
    inputTensor = Input(shape = (ntimeSteps, nchannels), name = 'input')

    previousTensor = inputTensor #previousTensor is used to link all of the tensors together

    #temporary while waiting for skimage to be installed on oscer
    previousTensor=AveragePooling1D(
                pool_size=2, 
                strides = 2,
                name='AdvPool'
            )(previousTensor)

    count=0 #used to name layers
    #Creates the convolution layers
    for i in convLayers:
        name = 'C%02d'%count
        previousTensor = Convolution1D(
            filters = i['filters'],
            kernel_size = i['kernelSize'],
            strides = i['kernelStrides'],
            padding = 'valid',
            use_bias = True,
            kernel_initializer = 'random_uniform',
            bias_initializer = 'zeros',
            name = name,
            activation = 'elu',
            kernel_regularizer=regularizer
        )(previousTensor)

        if(i['poolSize'] > 1):
            name='Max%02d'%count
            previousTensor=MaxPooling1D(
                pool_size=i['poolSize'], 
                strides = i['poolStrides'],
                name=name
            )(previousTensor)
        count+=1

    previousTensor = Flatten()(previousTensor)

    count = 0#used to name layers
    #Creates the dense layers
    for i in denseLayers:
        name = 'D%02d'%count
        previousTensor = Dense(
            units = i['units'],
            activation = 'elu',
            use_bias = 'True',
            bias_initializer = 'zeros',
            name = name,
            kernel_regularizer=regularizer
        )(previousTensor)

        if pDropout > 0:
            name='Drop%02d'%count
            previousTensor=Dropout(pDropout, name=name)(previousTensor)
        count+=1

    output = Dense(
        units = nclasses,
        activation = 'softmax',
        bias_initializer = 'zeros',
        name = 'output'
    )(previousTensor)

    model = Model(inputs = inputTensor, outputs = output) #Actually creates the model
    opt = tf.keras.optimizers.Adam(lr=.0001, beta_1=.9, beta_2=.999, epsilon=None, decay=0.0, amsgrad=False) #Creates the optimizer
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['categorical_accuracy']) #Compiles the model with the optimizer and metrics
    print(model.summary())
    return model