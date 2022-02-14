import fnmatch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import heapq
import sklearn
import metrics

#Loads the results matching fileBase in dirName
def loadResults(dirName, fileBase):
    files = fnmatch.filter(os.listdir(dirName), fileBase)
    files.sort()
    
    results = []
    for f in files:
        results.append(pickle.load(open("%s/%s"%(dirName, f), "rb")))

    return results

#Displays graphs of the valiation and training metric over time
#Also returns the average value of metric in each        
def visualizeExperiment(dirName, fileBase, metric='categorical_accuracy'):
    results = loadResults(dirName, fileBase)

    for i, r in enumerate(results):
        plt.plot(r['history'][metric], label='Model {:d}'.format(i+1))
    plt.title('Training')
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend(loc='lower right', prop={'size': 10})
    plt.show()

    for i, r in enumerate(results):
        plt.plot(r['history']['val_' + metric], label='Model {:d}'.format(i+1))
    plt.title('Validation')
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend(loc='lower right', prop={'size': 10})
    plt.show()

    accuracy = 0
    for r in results:
        #uses the average of the top 10 accuracies in a result as its accuracy
        accuracy += np.average(heapq.nlargest(10, r['history']['val_' + metric]))
    print('Average Val Accuracy: ', (accuracy/len(results)))

#Displays the confusion matrix
def visualizeConfusion(dirName, fileBase, key_true='true_validation', key_predict='predict_validation'):
    results = loadResults(dirName, fileBase)

    for r in results:
        preds = r[key_predict]
        trues = r[key_true]
        metrics.generate_confusion_matrix(trues, preds, ['28', '29', '30', '31', '32', '33'])


if __name__=='__main__':
    visualizeExperiment('results', '*.pkl')
    visualizeConfusion('results', '*.pkl')