# Undergradute Honor's Thesis
The purpose of this experiment is to automaticallly predict the Master of Propulsion MOCS scores that are generated for the SIPPC based on the kinematics data that the SIPPC records using a custom, 1D CNN

## Setup
### Creating the Data CSV
To prepare the data, first all of the trials that are going to be analyzed must be entered into a CSV. <br>
The CSV's header must be `Subject,Date,Trial,File,0,30,60,90,120,150,180,210,240,270,FinalScore` <br>
Where <br>
* Subject if the subject number
* Date is the date of the trial in MM/DD format
* Trial is the trial number recorded
* File is the kinematic mat file that stores the kinematic data for the trial
    * It is important that the entry is only recorded if the kinematic data is considered 'ok' or better
* 0,30,60,90,120,150,210,240,270 are the scores for each 30 second interval
* Final score is the final score that the grader calculated and is used to verify data integrity
<br>
An example of this is as follows
```
Subject,Date,Trial,File,0,30,60,90,120,150,180,210,240,270,FinalScore
29,01/22,1,kin_file_name.mat,31,28,28,32,31,31,32,31,31,31,14
```
Here, this was subject 29, the trial took place on January 22nd, trial 1 was scored, and the baby got a score of 31 for the first 30 seconds, 28 for the seconds, etc. and the grader found that the baby's final score was 14.

### Preparing the Kinematic Data
Once the CSV is created, each of the mat files recorded need to be saved as a pkl file. This can be done using `pklMat.py` <br>
To do this:
1. Move all of the needed mat files into a folder
2. Call `python3 pklMat.py <CSV_DATAFILE.csv> <DIRECTORY WITH MAT FILES> <DIRECTORY TO PUT THE CSV FILES>` 
    *the directory to put the csv files is optional and will default to using the same directory that the mat files are in.

## Training the Model
The current version of the model is setup to use the Bayesian Hyperparameter tuner. If you prefer to handtune the hyperparameters, this can be done with the code in the `noTuner` folder.

### Starting the Tuner
To start running the tuner, call `python3 tunerExperiment.py` with the proper parameters which are as follows:

#### exp
This is put directly in the name of the tuner and models that get trained to help identify which is which. <br>
Defaults to 'Propulsion'

#### rot
Specifies which rotation of the data is used <br>
Defaults to 1

#### foldsPath
The path to the directory that contains the csv files with each fold <br>
Defaults to 'folds'

#### resultsPath
The path to the directory to store the final outputted model in. <br>
Defaults to 'results'

#### logDir
The path to the directory to store the tuner logs in <br>
Defaults to 'logs'

#### pklDir
The path to the directory with the kinematic pkl files generated in the Preparing the Kinematic Data section <br>
Defaults to ''

#### trials
The number of trials to run the tuner for <br>
Defaults to 10

#### overwrite
The tuner's name is generated based on the resultsPath, the exp and the trials. <br>
If overwrite is set to 0, then if a tuner with the same exp, resultsPath and trials has already been created then the program will continue from where that tuner left off. <br>
If overwrite is set to 1, then if a tuner with the same exp, resultsPath and trials has already been created then the program will overwrite it and start over. <br>
Defaults to 0

#### tune 
If this is 0, the program will just use best hyperparameteres that the tuner has previously found and create a model based off of it <br>
If this is 1, the program will continue to tune the hyperparameters until it reaches the specified number of trials <br>
Defaults to 1

#### epochs
This is the number of epochs to train the final model for. Not used for tuning <br>
Defaults to 10

#### patience
The patience the final model will use. Not used for tuning <br>
Defaults to 100

### Training the Model
To actually train the model based on the best set of hyperparameters found so far, run `python3 -exp <EXP_NAME_OF_TUNER> -trials <TRIALS_SET_FOR_THE_TUNER> -tune 0`. <br>
This will load in the tuner that used the same exp and trials specified and created a model based off of it. <br>
It is a good idea to change the number of epochs, patience and rot when doing this! <br>
<br>
A model will also automatically be trained when a tuner reaches the specified number of trials based on the best hyperparameters that the tuner found.

### Changing the Tuner
I did not make this doable from the commandline. To change how the tuner itself runs, you must go to tunerModel.py and change the code in `build_model`

## Analysis
### Visualize Model
Takes three arguments:
* dirName: the directory that contains the results of interest
* fileBase: the regex for the files of interest
* metric: The metric of interest. Defaults to categorical_accuracy
This will create plots showing how the models progressed on the training data and the validation data

### Visualize Confusion
Takes four arguments:
* dirName: the directory that contains the results of interest
* fileBase: the regex for the files of interest
* types: list containing 'validation' for the validation data and/or 'test' for the test data. Defaults to \['validation'\]
* plot: if the function should show the plot of the confusion matrix. Defaults to True
Creates a confusion matrix of the specified files with the specified types and returns it. Will also display a plot of the confusion matrix if plot is True

### getFullResults
Takes 5 arguments:
* dirName: the directory that contains the results of interest
* argBase: the regex for the args of interest
* fileBase: the regex for the models of interest. Should be the same as argBase except '_results.pkl' should be replaced with '_model'
* split: either 'rot' or 'cp' if you're curious in comparing models based on rotation or if the baby was at risk of cerebral palsy or not. Defaults to ;rot'
* valid: True if comparing validation data. False if comparing test data. Defaults to False.
This loads in the specified models and outputs various statistics based on the 5 minute mastery of propulsion MOCS scores that the model outputted vs. the true 5 minute mastery of propulsion MOCS scores.
