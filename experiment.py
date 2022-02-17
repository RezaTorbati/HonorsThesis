import pandas as pd
import numpy as np
import argparse
import pickle
import random
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow import keras

from prep import parsePropulsionCSV
from model import create_model

#Loads in and prepares the data
def load_data(args):
    ins, outs = parsePropulsionCSV(args.trainCSVDir, args.trainCSV, args.pklDir)
    
    #Removes any bad intervals and subtracts 28 from the value
    i=0
    while i < len(outs):
        if outs[i] == -1:
            del(outs[i])
            del(ins[i])
            i-=1
        else:
            outs[i] -= 28
        i+=1     
    #Converts the ins and outs from a list to a numpy array   
    ins = np.array(ins)
    outs = np.array(outs)

    #Repeats above but for validation data
    validIns, validOuts = parsePropulsionCSV(args.validCSVDir, args.validCSV,args.pklDir)
    i=0
    while i < len(validOuts):
        if validOuts[i] == -1:
            del(validOuts[i])
            del(validIns[i])
            i-=1
        else:
            validOuts[i] -= 28
        i+=1
    validIns = np.array(validIns)
    validOuts = np.array(validOuts)

    #1hot encodes the outputs
    outs = np.eye(args.nclasses)[outs]
    validOuts = np.eye(args.nclasses)[validOuts]

    return ins, outs, validIns, validOuts

#Creates a string with all of the important training metadata to be used for file names
def generate_fname(args):
    #conv layers
    filters = '_'.join(str(x) for x in args.filters)
    kernelSizes = '_'.join(str(x) for x in args.kernel_sizes)
    kernelStrides = '_'.join(str(x) for x in args.kernel_strides)
    pools = '_'.join(str(x) for x in args.pool_sizes)
    poolStrides = '_'.join(str(x) for x in args.pool_strides)

    #dense layers
    dense = '_'.join(str(x) for x in args.dense)

    # Dropout
    if args.dropout is None:
        dropout = ''
    else:
        dropout = '_drop_%0.2f'%(args.dropout)

    # l2
    if args.l2 is None:
        l2 = ''
    else:
        l2 = '_l2_%0.3f'%(args.l2)

    return '%s/%s_filters_%s_kernels_%s_kernelStrides_%s_pools_%s_poolStrides_%s_dense_%s%s_%s'%(
        args.resultsPath,
        args.exp,
        filters,
        kernelSizes,
        kernelStrides,
        pools,
        poolStrides,
        dense,
        dropout,
        l2
    )

#Used to generate batches of examples
#inputName needs to match the name of the input layer
#outputName needs to match the name of the output layer
def batch_generator(ins, outs, batchSize, inputName='input', outputName='output'):
    while True:
        #Gets a batchSize sized sample from the inputs
        indicies = random.choices(range(ins.shape[0]), k=batchSize)

        #Returns a list of the selected examples and their corresponding outputs
        yield({inputName: ins[indicies,:,:]}, {outputName: outs[indicies,:]})

def execute_exp(args):
    ins, outs, validIns, validOuts = load_data(args)
    timeSteps = len(ins[0])
    channels = len(ins[0][0])

    dense_layers = [{'units': i} for i in args.dense]
    conv_layers = [{'filters': f, 'kernelSize': k, 'kernelStrides': ks, 'poolSize': p, 'poolStrides': ps}
                for f, k, ks, p, ps in zip(args.filters, args.kernel_sizes, args.kernel_strides, args.pool_sizes, args.pool_strides)]

    #Creates the model
    model = create_model(timeSteps,
        channels, 
        nclasses=args.nclasses,
        denseLayers=dense_layers,
        convLayers=conv_layers,
        pDropout=args.dropout,
        l2=args.l2)

    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)

    #batch generator
    generator = batch_generator(ins, outs, batchSize=args.batchSize)

    if args.batchSize is not None:
        #Runs the model
        history = model.fit(x=generator, epochs=args.epochs,
            steps_per_epoch = args.stepsPerEpoch,
            verbose=True,
            validation_data=(validIns, validOuts),
            callbacks=[early_stopping_cb])
    else:   
        #Runs the model
        history = model.fit(x=ins, y=outs, epochs=args.epochs,
            steps_per_epoch = args.stepsPerEpoch,
    	    verbose=True,
    	    validation_data=(validIns, validOuts),
    	    callbacks=[early_stopping_cb])

    # Generate log data
    results = {}
    results['args'] = args
    results['predict_training'] = model.predict(ins)
    results['predict_training_eval'] = model.evaluate(ins, outs)
    results['true_training'] = outs
    results['predict_validation'] = model.predict(validIns)
    results['predict_validation_eval'] = model.evaluate(validIns, validOuts)
    results['true_validation'] = validOuts
    #results['predict_testing'] = model.predict(ins_testing)
    #results['predict_testing_eval'] = model.evaluate(ins_testing, outs_testing)
    #results['folds'] = folds
    results['history'] = history.history

    # Save results
    fbase = generate_fname(args)
    results['fname_base'] = fbase
    fp = open("%s_results.pkl"%(fbase), "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Model
    model.save("%s_model"%(fbase))
    
    return model

#Create the parser for the command-line arguments
def create_parser():
    parser = argparse.ArgumentParser(description='Mastery of Propulsion Learner', fromfile_prefix_chars='@')
    
    parser.add_argument('-exp', type=str, default='Propulsion', help='Tag to be put in file name')
    parser.add_argument('-resultsPath', type=str, default='results', help='Directory to store results in')
    parser.add_argument('-trainCSV', type=str, default='MasteryOfPropulsionTrain.csv', help='Training mastery of propulsion csv file')
    parser.add_argument('-trainCSVDir', type=str, default='.', help='Training mastery of propulsion csv directory')
    parser.add_argument('-validCSV', type=str, default='MasteryOfPropulsionValid.csv', help='Validation mastery of propulsion csv file')
    parser.add_argument('-validCSVDir', type=str, default='.', help='Validation mastery of propulsion csv directory')
    parser.add_argument('-pklDir', type=str, default='', help='Directory to the pkl files')
    parser.add_argument('-nclasses',type = int, default = 6, help='Number of output classes')
    
    parser.add_argument('-batchSize', type=int, default=None, help='training batch size')
    parser.add_argument('-epochs',type=int, default = 10, help='Number of epochs to run for')
    parser.add_argument('-stepsPerEpoch', type=int, default=None, help='Steps taken per epoch')
    parser.add_argument('-patience', type=int, default = 100, help='Patience for early termination')
    parser.add_argument('-min_delta', type=float, default = 0.0, help='Min delta for early termination')

    parser.add_argument('-kernel_sizes', nargs='+', type=int, default=[20,10], help='Kernel Sizes')
    parser.add_argument('-filters', nargs='+', type=int, default=[75,100], help='Filter Sizes')
    parser.add_argument('-kernel_strides', nargs='+', type=int, default=[5,1], help='Kernel Strides')
    parser.add_argument('-pool_sizes', nargs='+', type=int, default=[1,5], help='Pooling sizes')
    parser.add_argument('-pool_strides', nargs='+', type = int, default=[1,5], help = 'pooling strides')
    parser.add_argument('-l2', type=float, default=0, help='Amount of l2 regularization')

    parser.add_argument('-dense', nargs='+', type=int, default = [1000,100], help='Size of the dense layers')
    parser.add_argument('-dropout', type=float, default=0, help='dropout rate')

    return parser

if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(12)
    parser = create_parser()
    args = parser.parse_args()

    execute_exp(args)
    
