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
An example of this is as follows
```
Subject,Date,Trial,File,0,30,60,90,120,150,180,210,240,270,FinalScore
29,01/22,1,kin_stat1_2016Y_01M_22D_12h_19m_29s_S3-subject-29_trials_1.mat,31,28,28,32,31,31,32,31,31,31,14
```
Here, this was subject 29, the trial took place on January 22nd, trial 1 was scored, and the baby got a score of 31 for the first 30 seconds, 28 for the seconds, etc. and the grader found that the baby's final score was 14.

### Preparing the Kinematic Data
Once the CSV is created, each of the mat files recorded need to be saved as a pkl file. This can be done using `pklMat.py` <br>
To do this:
1. Move all of the needed mat files into a folder
2. Call `python3 pklMat.py <CSV_DATAFILE.csv> <DIRECTORY WITH MAT FILES> <DIRECTORY TO PUT THE CSV FILES>` where the directory to put the csv files is optional and will default to using the same directory that the mat files are in.

## Training the Model
The current version of the model is setup to use the Bayesian Hyperparameter tuner. If you prefer to handtune the hyperparameters, this can be done with the code in the `noTuner` folder.


## Analysis
