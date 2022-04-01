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
    test = '%s/rot%d'%(args.rotsPath, args.rot)

    ins, outs = parsePropulsionCSV(args.trainCSVDir, args.trainCSV, args.pklDir, args.reduce)
    
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
    validIns, validOuts = parsePropulsionCSV(args.validCSVDir, args.validCSV,args.pklDir, args.reduce)
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

    testIns=None
    testOuts = None
    if args.testCSV is not None:
        testIns, testOuts = parsePropulsionCSV(args.testCSVDir, args.testCSV,args.pklDir, args.reduce)
        i=0
        while i < len(testOuts):
            if testOuts[i] == -1:
                del(testOuts[i])
                del(testIns[i])
                i-=1
            else:
                testOuts[i] -= 28
            i+=1
        testIns = np.array(testIns)
        testOuts = np.array(testOuts)

    #1hot encodes the outputs
    outs = np.eye(args.nclasses)[outs]
    validOuts = np.eye(args.nclasses)[validOuts]
    if testOuts is not None:
        testOuts = np.eye(args.nclasses)[testOuts]

    return ins, outs, validIns, validOuts, testIns, testOuts

#Creates a string with all of the important training metadata to be used for file names
def generate_fname(args):
    #conv layers
    filters = '_'.join(str(x) for x in args.filters)
    kernelSizes = '_'.join(str(x) for x in args.kernel_sizes)
    pools = '_'.join(str(x) for x in args.pool_sizes)
    poolStrides = '_'.join(str(x) for x in args.pool_strides)

    #dense layers
    dense = '_'.join(str(x) for x in args.dense)

    dropout = '_drop_%.2f'%(args.dropout)
    l2 = '_l2_%.5f'%(args.l2)
    sDropout = '_sDrop_%.2f'%(args.sDropout)

    return '%s/%s_r%s_f%s_k%s_p%s_pStrides%s_d%s%s%s%s'%(
        args.resultsPath,
        args.exp,
        args.reduce,
        filters,
        kernelSizes,
        pools,
        poolStrides,
        dense,
        dropout,
        sDropout,
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
    ins, outs, validIns, validOuts, testIns, testOuts = load_data(args)
    timeSteps = len(ins[0])
    channels = len(ins[0][0])

    dense_layers = [{'units': i} for i in args.dense]
    conv_layers = [{'filters': f, 'kernelSize': k, 'poolSize': p, 'poolStrides': ps}
                for f, k, p, ps in zip(args.filters, args.kernel_sizes, args.pool_sizes, args.pool_strides)]

    #Creates the model
    model = create_model(timeSteps,
        channels, 
        nclasses=args.nclasses,
        denseLayers=dense_layers,
        convLayers=conv_layers,
        pDropout=args.dropout,
        l2=args.l2,
        sDropout=args.sDropout)

    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      monitor='val_categorical_accuracy',
                                                      mode='max',
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

    if args.testCSV is not None:
        results['predict_testing'] = model.predict(testIns)
        results['predict_testing_eval'] = model.evaluate(testIns, testOuts)
        results['true_testing'] = testOuts

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
    parser.add_argument('-rot', type=int, default=1, help='rotation')
    parser.add_argument('-rotsPath', type=str, default='rotations', help='Path to the csv files with the rotations')
    parser.add_argument('-resultsPath', type=str, default='results', help='Directory to store results in')
    #parser.add_argument('-trainCSV', type=str, default='MasteryOfPropulsionTrain.csv', help='Training mastery of propulsion csv file')
    #parser.add_argument('-trainCSVDir', type=str, default='.', help='Training mastery of propulsion csv directory')
    #parser.add_argument('-validCSV', type=str, default='MasteryOfPropulsionValid.csv', help='Validation mastery of propulsion csv file')
    #parser.add_argument('-validCSVDir', type=str, default='.', help='Validation mastery of propulsion csv directory')
    #parser.add_argument('-testCSV', type=str, default=None, help='Test mastery of propulsion csv file')
    #parser.add_argument('-testCSVDir', type=str, default='.', help='Test mastery of propulsion csv directory')
    parser.add_argument('-pklDir', type=str, default='', help='Directory to the pkl files')
    parser.add_argument('-reduce', type=int, default=1, help='amount to initially reduce the array by')
    parser.add_argument('-nclasses',type = int, default = 6, help='Number of output classes')
    
    parser.add_argument('-batchSize', type=int, default=None, help='training batch size')
    parser.add_argument('-epochs',type=int, default = 10, help='Number of epochs to run for')
    parser.add_argument('-stepsPerEpoch', type=int, default=None, help='Steps taken per epoch')
    parser.add_argument('-patience', type=int, default = 100, help='Patience for early termination')
    parser.add_argument('-min_delta', type=float, default = 0.0, help='Min delta for early termination')

    parser.add_argument('-kernel_sizes', nargs='+', type=int, default=[20,10], help='Kernel Sizes')
    parser.add_argument('-filters', nargs='+', type=int, default=[75,100], help='Filter Sizes')
    parser.add_argument('-pool_sizes', nargs='+', type=int, default=[1,5], help='Pooling sizes')
    parser.add_argument('-pool_strides', nargs='+', type = int, default=[1,5], help = 'pooling strides')
    parser.add_argument('-l2', type=float, default=0, help='Amount of l2 regularization')

    parser.add_argument('-dense', nargs='+', type=int, default = [1000,100], help='Size of the dense layers')
    parser.add_argument('-dropout', type=float, default=0, help='dropout rate for dense layers')
    parser.add_argument('-sDropout', type=float, default=0, help='dropout rate for filters')

    return parser

if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(12)
    parser = create_parser()
    args = parser.parse_args()

    execute_exp(args)
    
