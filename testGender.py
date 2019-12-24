#IMPORT REQUIRED LIBRARIES

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as MFCC
from sklearn import preprocessing
import warnings

#GENERAL WARNING FILTER IGNORES MATCHING WARNINGS
warnings.filterwarnings("ignore")

#TO GET MEL FREEQUENCY CEPSTRAL COEFFICIENTS OF THE AUDIO SIGNAL
def get_MFCC(sr,audio):
    
    #MFCC IS SUBSTITUE FOR python_speech_features CLASS. THIS CREATES A SHAPE FOR THE AUDIO 
    features = MFCC.mfcc(audio,sr,0.025,0.01,13,appendEnergy=False)
    
    #feat IS THE STORAGE FOR SIGNAL FEATURES OF GIVEN AUDIO SIGNAL
    feat = np.asarray(())

    #SHAPE RETURNS THE DIMENSIONS OF THE OBJECT features
    for i in range(features.shape[0]):
        
        #temp is TEMPORARY STOREAGE FOR features LIST'S ELEMENTS FROM i TO END
        temp = features[i,:]

        #IF THE temp MIN VALUE IS NAN, FIND A VALID RANGE
        if np.isnan(np.min(temp)):
            continue
        else:
            
            #WHEN feat IS EMPTY, INITIALIZE feat
            if feat.size == 0 :
                feat = temp
            else:
                
                #WHEN feat IS ALREADY INITIALIZED, APPEND FURTHER VALUES
                feat = np.vstack((feat,temp))
    features = feat
    
    #TO STANDARDIZE ANY DATASET ALONG ANY AXIS
    features = preprocessing.scale(features)
    
    #RETURNS STANDARDIZED DATA    
    return features                

#SOURCE PATH TO FETCH AUDIO FILE FOR TESTING
sourcePath = "/home/siva/GenderClassification/Dataset/test_data/AudioSet/female_clips"

#LOCATION TO FETCH ALL MODULES FOR TESTING i.e .gmm file's Location.
modelPath = "/home/siva/GenderClassification/Modules/"

#TO COLLECT ALL .gmm FILES
gmmFiles = [os.path.join(modelPath,fname) for fname in os.listdir(modelPath) if fname.endswith('.gmm')]

#COLLECT MODULE NAMES FROM .gmm FILES
models = [cPickle.load(open(fname,'rb')) for fname in gmmFiles]

#LIST OF ALL CLASSIFICATIONS. IN THIS CASE, THE LIST STORES ONLY MALE AND FEMALE
genders = [fname.split("/")[-1].split(".gmm")[0] for fname in gmmFiles]

#COLLECT ALL TESTING FILES
files = [os.path.join(sourcePath,f) for f in os.listdir(sourcePath) if f.endswith(".wav")]

#FOR ALL COLLECTED FILES, SPECIFY THE GENDER.
for f in files:

    #TO PRINT FILE NAME
    print(f.split("/")[-1])
    
    #TO RETRIVE SAMPLING RATE AND AUDIO SIGNAL
    sr,audio = read(f)

    #features STORES THE RETRUNED VALUE FROM THE get_MFCC() FUNCTION
    features = get_MFCC(sr,audio)
    
    #EMPTY LIST TO STORE SCORES
    scores = None

    #log_likelihood IS ANY ATTRIBUTES THAT ARE DIRECTLY PROPORTIONAL
    #CREATE AN ARRAY OF SIZE OF MODELS, SET TO 0
    log_likelihood = np.zeros(len(models))

    #TO CHECK EVERY MODEL
    for i in range(len(models)):
        
        gmm = models[i]
        
        #score() COMPUTES THE per-sample average log-likelihood OF THE GIVEN DATA
        scores = np.array(gmm.score(features))
        
        #sum() SUMS ALL THE SCORES FOR TAKEN MODEL
        log_likelihood[i] = scores.sum()
    
    #GENDER IS DECIDED BASED ON THE MAXIMUM SCORES ATTAINED    
    winner = np.argmax(log_likelihood)
    
    print(genders[winner])

    #THEIR CORRESPONDING SCORES CAN BE RETRIVED FROM log_likelihood
    #FOR FEMALE USE log_likelihood[0]
    #FOR MALE USE log_likelihood[1]
    #BY THIS MODEL, THE ACCURACY FOR FEMALE VOICE IS 96% WHERE AS ACCURACY FOR MALE VOICE IS 76% 