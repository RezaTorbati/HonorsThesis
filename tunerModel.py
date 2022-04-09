import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, BatchNormalization, Dropout, AveragePooling1D, SpatialDropout1D
from tensorflow.keras import Model
import keras_tuner as kt

def build_model(hp):
    dropout = hp.Float('dropout', 0, .5, step=.05)
    sDropout = hp.Float('sparseDropout', 0, .25, step=.05)
    activation = hp.Choice('activaion', values=['elu', 'sigmoid', 'exponential', 'selu'])
    #convActivation = hp.Choice('convActivation', values=['elu', 'sigmoid', 'exponential', 'selu'])

    l2 = hp.Choice('l2', values=[1e-4, 1e-5, 1e-6, 0.0])
    l1 = hp.Choice('l1', values=[1e-4, 1e-5, 1e-6, 0.0])
    regularizer = None
    if(l1 > 0 and l2 <= 0):
        regularizer = keras.regularizers.l1(l1)
    elif(l2 > 0 and l1 <= 0):
        regularizer = keras.regularizers.l2(l2)
    elif(l1 > 0 and l2 > 0):
        regularizer = keras.regularizers.l1_l2(l1, l2)

    #I assume 3 dense layers
    dense1 = hp.Int('dense1', 24, 90, step=6)
    dense2 = hp.Int('dense2', 24, 90, step=6)
    dense3 = hp.Int('dense3', 12, 30, step=6)
    dense_layers = [dense1, dense2, dense3]

    learningRate = hp.Choice('learningRate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    reduction = hp.Choice('reduction', values = [1,2,4,6]) 

    #Sets up the kernel sizes, the pool sizes and the pool strides
    #these will all reduce the network down to one timestep before flattening
    if reduction == 4:
        kernelSizes = [5,8,8,8,8,8,8,8,8,8,7,7,7,6,5]
        poolSizes =   [1,1,1,3,1,1,3,1,1,3,1,1,1,1,1]
        poolStrides = [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1]
    elif reduction == 1:
        kernelSizes = [5,8,8,8,8,8,8,8,8,8,8,8,7,7,6]
        poolSizes =   [1,1,1,5,1,1,4,1,1,4,1,1,1,1,1]
        poolStrides = [1,1,1,4,1,1,3,1,1,3,1,1,1,1,1]
    elif reduction == 2:
        kernelSizes = [5,8,8,8,8,8,8,8,7,7,7,7,7,6,5]
        poolSizes =   [1,1,1,4,1,1,4,1,1,3,1,1,1,1,1]
        poolStrides = [1,1,1,3,1,1,3,1,1,2,1,1,1,1,1]
    elif reduction == 6:
        kernelSizes = [5,8,8,8,8,8,8,7,7,7,7,7,7,6,5]
        poolSizes =   [1,1,1,3,1,1,3,1,1,2,1,1,1,1,1]
        poolStrides = [1,1,1,2,1,1,2,1,1,1,1,1,1,1,1]

    #sets up the filters
    f1 = hp.Int('f1', 10,25,step=5)
    f2 = hp.Int('f2', 5,25,step=5)
    f3 = hp.Int('f3', 15,30,step=5)
    f4 = hp.Int('f4', 10,30,step=5)
    f5 = hp.Int('f5', 20,35,step=5)
    f6 = hp.Int('f6', 15,35,step=5)
    f7 = hp.Int('f7', 25,40,step=5)
    f8 = hp.Int('f8', 20,40,step=5)
    f9 = hp.Int('f9', 20,40,step=5)
    f10 = hp.Int('f10', 25,50,step=5)
    f11 = hp.Int('f11', 35,90,step=5)
    filters = [f1,f2,f2,f3,f4,f4,f5,f6,f6,f7,f8,f8,f9,f10,f11]
   
    conv_layers = [{'filters': f, 'kernelSize': k, 'poolSize': p, 'poolStrides': ps}
            for f, k, p, ps in zip(filters, kernelSizes, poolSizes, poolStrides)]

    #I'm just going to hardcode these because they probably aren't changing
    nclasses=6 
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

        if(i['poolSize'] > 1):
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
    #print(model.summary())
    return model