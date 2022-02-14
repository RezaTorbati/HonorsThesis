import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Model

def create_model(ntimeSteps, nchannels, nclasses, convLayers, denseLayers, pDropout=0):
    inputTensor = Input(shape = (ntimeSteps, nchannels), name = 'input')

    previousTensor = inputTensor

    count=0 #used to name layers
    for i in convLayers:
        name = 'C%02d'%count
        previousTensor = Convolution1D(
            filters = i['filters'],
            kernel_size = i['kernelSize'],
            strides = i['kernelStrides'],
            padding = 'same',
            use_bias = True,
            kernel_initializer = 'random_uniform',
            bias_initializer = 'zeros',
            name = name,
            activation = 'elu'
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
    for i in denseLayers:
        name = 'D%02d'%count
        previousTensor = Dense(
            units = i['units'],
            activation = 'elu',
            use_bias = 'True',
            bias_initializer = 'zeros',
            name = name
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

    model = Model(inputs = inputTensor, outputs = output)
    opt = tf.keras.optimizers.Adam(lr=.0001, beta_1=.9, beta_2=.999, epsilon=None, decay=0.0, amsgrad=False)        
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['categorical_accuracy'])
    print(model.summary())
    return model