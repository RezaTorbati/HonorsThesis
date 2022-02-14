import numpy as np
import h5py
import os
import sys
import pickle
import pandas as pd

#Converts a single kin mat file
def convertMat(fname, target = ''):  
    if target == '':
        target = os.path.splitext(fname)[0]+'.pkl'

    h5 = h5py.File(fname,'r')

    trials = []

    #Loops through each trial
    for trial in h5.get('state/trial_stat1'):
        t = h5[trial[0]] #IDK why but trial is an array of length one so must specify the 0th index
        outterDict = {} #Stores things like 'back' or 'r_wrist' as keys and innderDict as the data
        for i in t.keys():
            innerDict = {} #Stores things like 'path_length' or 'velocity' as keys and the actual values as data
            for j in t.get(i).keys():
                #print(i, '    ', j)
                data = np.array(t.get(i).get(j)) #Gets the list of actual values corresponding to j
                innerDict[j] = data
            outterDict[i] = innerDict
            #print()
        #print('\n')
            
        trials.append(outterDict)

    pickle.dump(trials, open(target, 'wb'))

#Converts all of the kin files inside of a master of propulsion mat
def convertCSV(csvName, directory, outDirectory =  ''):
    if outDirectory == '':
        outDirectory = directory

    df = pd.read_csv(csvName)

    for index, row in df.iterrows():
        src = directory + '/' + row['File']

        target = outDirectory + '/' + row['File']
        target = os.path.splitext(target)[0]+'.pkl'

        convertMat(src, target)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Expects csv file and directory as arguments")
        print("Also, optional 4th arg for out directory")
        exit(-1)

    if(sys.argv[3] is not None):
        convertCSV(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        convertCSV(sys.argv[1], sys.argv[2])