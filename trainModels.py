#IMPORT REQUIRED LIBRARIES

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
import python_speech_features as MFCC
from sklearn import preprocessing
import warnings

#GENERAL WARNING FILTER IGNORES MATCHING WARNINGS
warnings.filterwarnings("ignore")

#TO GET MEL FREEQUENCY CEPSTRAL COEFFICIENTS OF THE AUDIO
def get_MFCC(sr,audio):
    
    #MFCC IS SUBSTITUE FOR python_speech_features CLASS. THIS CREATES A SHAPE FOR THE AUDIO 
    features = MFCC.mfcc( audio , sr , 0.025 , 0.01 , 13 , appendEnergy = False )
    
    #TO STANDARDIZE ANY DATASET ALONG ANY AXIS
    features = preprocessing.scale(features)

    #RETURNS STANDARDIZED DATA
    return features

#SOURCES OF AUDIO TRAINING FILE
#FOR DIFFERENT MODULES CLASSIFY EACH INTO SEPERATE LOCATION AND PROVIDE THE DATA ACCORDINGLY
#HERE ONLY TWO MODULES SUCH AS MALE AND FEMALE ARE USED
#RUN THIS PROGRAM WITH SOURCE DENOTING MALE AUDIO TRAININGS ONCE AND FEMALE AUDIO TRAININGS ONCE
#THIS CREATES TWO DIFFERENT MODULES IN THE GIVEN DESTINATION
#ONE IS MALE VOICE MODULE AND OTHER IS FEMALE VOICE MODULE

source = "/home/siva/GenderClassification/Dataset/train_data/youtube/male"

#DESTINATION LOCATION TO STORE MODULES
dest = "/home/siva/GenderClassification/Modules/"

#COLLECT ALL THE FILES WHICH ARE TRAINING MODULE. HERE THE AUDIO FILE IS EXPECTED TO BE IN .wav FORMAT
# files IS A LIST OF ONLY .wav FILES FROM THE PROVIDED SOURCE LOCATION
files = [os.path.join(source,f) for f in os.listdir(source) if f.endswith('.wav')]

#TO STORE RETURNED DATA FROM get_MFCC() FUNCTION
features = np.asarray(())

#TO READ ALL TRAIN FILES, RUN A FOR LOOP ON THE COLLECTION
for f in files:
    
    #sr DENOTES SAMPLING RATE AND audio DENOTES THE SOURCE SIGNAL
    sr,audio = read(f)
    
    #vector STORES THE RETRUNED VALUE FROM THE get_MFCC() FUNCTION
    vector  = get_MFCC(sr,audio)

    #FOR THE FIRST AUDIO SIGNAL features REMAINS EMPTY. SO ASSIGN THE RETURNED VALUE TO features
    if features.size == 0:
        features = vector
    else:
        #FOR THE FOLLOWING ITERATIONS DEVELOP features AS A VECTOR STACK 
        features = np.vstack((features, vector)) 

#ONCE ALL TRAINING FILES ARE STACKED, CREATE A GAUSSIAN MIXTURE MODEL
gmm = GaussianMixture(n_components = 8, covariance_type='diag', max_iter = 200 , n_init = 3 )

#fit() ESTIMATES THE MODEL PARAMETERS USING THE EM ALGORITHM
gmm.fit(features)

#picklefile PROVIDES THE DESTINATION TO STORE THE .gmm FILE
pickleFile = f.split("/")[-2].split(".wav")[0]+".gmm"

#MODEL SAVED IN FILE BY dump() FUNCTION
cPickle.dump(gmm,open(dest + pickleFile,'wb'))

print ("Modeling Completed for Gender : " + pickleFile)