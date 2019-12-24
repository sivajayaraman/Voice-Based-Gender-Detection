# Voice Based Gender Detection using Machine Learning

This machine learning program detects gender based on provided voice input. This is built in two phases. The first phase is the training phase where the module is trained with all possible classes or models. Classes or Models are all possible outcomes for a particular input. In our case, the possible outcomes are, "MALE" and "FEMALE". These classes are developed with all properties of the training data. This develops a general pattern for the classes denoting that, If this is the voices' gender, then these are the properties expected to be satisfied. Once the training is done, This module can be tested.

The second phase is to test the module. This can be done by providing a testing voice to detect the gender. This testing voice properties are extracted and compared with the properties of both classes. Then based on the scores, the dender is decided.

This is refered from https://appliedmachinelearning.blog/2017/06/14/voice-gender-detection-using-gmms-a-python-primer/

The Dataset is collected from https://www.dropbox.com/s/sqg7az7fja6rqfw/pygender.zip?dl=0
