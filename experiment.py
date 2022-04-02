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

def load_file(path, fname, args):
    ins, outs = parsePropulsionCSV(path, fname, args.pklDir, args.reduce)

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
    
    return ins, outs

#Loads in and prepares the data
def load_data(args):
    #I am assuming that there are 6 folds
    
    #Prepares the folds for the training, validation and test data
    train = []
    for i in range(0,6):
        train.append('fold%d.csv'%(i))
    
    test = 'fold%d.csv'%(args.rot)
    del train[args.rot]
    
    if args.rot > 0:
        validation = 'fold%d.csv'%(args.rot-1)
        del train[args.rot-1]
    else:
        validation = 'fold5.csv'
        del train[4] #This is 4 instead of 5 because I've already deleted an element above

    #Loads in the data
    print('Train data:')
    ins, outs = load_file(args.foldsPath, train[0], args)
    for fold in train[1:]:
        tmpIns, tmpOuts = load_file(args.foldsPath, fold, args)
        ins.extend(tmpIns)
        outs.extend(tmpOuts)
    
    print('Validation data:')
    validIns, validOuts = load_file(args.foldsPath, validation, args)
    
    print('Test data:')
    testIns, testOuts = load_file(args.foldsPath, test, args)

    #Converts the data from lists to numpy arrays   
    ins = np.array(ins)
    outs = np.array(outs)
    validIns = np.array(validIns)
    validOuts = np.array(validOuts)
    testIns = np.array(testIns)
    testOuts = np.array(testOuts)
    
    #One hot encodes the outputs
    outs = np.eye(args.nclasses)[outs]
    validOuts = np.eye(args.nclasses)[validOuts]
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

    return '%s/%s_r%s_f%s_k%s_p%s_pStrides%s_d%s%s%s%s_rot%d'%(
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
        l2,
        args.rot
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
    parser.add_argument('-foldsPath', type=str, default='folds', help='Path to the csv files with the folds')
    parser.add_argument('-resultsPath', type=str, default='results', help='Directory to store results in')
    parser.add_argument('-pklDir', type=str, default='', help='Directory to the pkl files')
    parser.add_argument('-reduce', type=int, default=1, help='amount to initially reduce the array by')
    parser.add_argument('-nclasses',type = int, default = 6, help='Number of output classes')
    
    parser.add_argument('-batchSize', type=int, default=None, help='training batch size')
    parser.add_argument('-epochs',type=int, default = 10, help='Number of epochs to run for')
    parser.add_argument('-stepsPerEpoch', type=int, default=None, help='Steps taken per epoch')
    parser.add_argument('-patience', type=int, default = 100, help='Patience for early termination')
    parser.add_argument('-min_delta', type=float, default = 0.0, help='Min delta for early termination')

    parser.add_argument('-kernel_sizes', nargs='+', type=int, default=[20,10], help='Kernel Sizes')
    parser.add_argument('-filters', nargs='+', type=int, default=[60,80], help='Filter Sizes')
    parser.add_argument('-pool_sizes', nargs='+', type=int, default=[5,1], help='Pooling sizes')
    parser.add_argument('-pool_strides', nargs='+', type = int, default=[5,1], help = 'pooling strides')
    parser.add_argument('-l2', type=float, default=0, help='Amount of l2 regularization')

    parser.add_argument('-dense', nargs='+', type=int, default = [100,20], help='Size of the dense layers')
    parser.add_argument('-dropout', type=float, default=0, help='dropout rate for dense layers')
    parser.add_argument('-sDropout', type=float, default=0, help='dropout rate for filters')

    return parser

if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(12)
    parser = create_parser()
    args = parser.parse_args()

    execute_exp(args)
    
