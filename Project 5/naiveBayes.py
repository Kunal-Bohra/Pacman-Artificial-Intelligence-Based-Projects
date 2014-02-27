# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'featureCounter' in this code refers to a counter of features
  (not to a raw samples.featureCounter).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

  def auxFunction(self, trainingData, trainingLabels):
    
    #Classify each datum using most probable label using
    #feature values of each pixel
    for r in range(self.dataSize):
      #Initialize the datum
      featureCounter = trainingData[r]
      #Initialize the feature value 
      featureCounterLabel = trainingLabels[r]
      #Increment for next feature counter label
      self.dataClass[featureCounterLabel]["occur"] += 1
      #Get features
      features = self.features
      #Loop for all extracted features
      for f in features:
        self.dataClass[featureCounterLabel][f] += featureCounter[f]
    return
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each featureCounter.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"

    #Initialize the variables
    self.dataSize = len(trainingData)
    #Size of training set
    self.dataClass = {}
    #Initialize class probability
    self.classProb = {}
    #Define valid labels
    labels = self.legalLabels
    #Loop for all legal labels 
    for label in labels:
      self.dataClass[label] = util.Counter()
      self.classProb[label] = {}
    #Set the training data  
    self.auxFunction(trainingData, trainingLabels)
    #Smoothing and calculating optmum value of k
    optimumK = 0
    #Occurence of optimum k
    oFreq = 0
    #Make grid
    grid = kgrid
    for k in grid:
      #Get labels 
      labels = self.legalLabels
      for l in labels:
        #Extract features
        features = self.features
        #Loop for all extracted features
        for f in features:
          #Smooth all current parameters estimates using formula from text
          #Use Laplace smoothing
          self.classProb[l][f] = (self.dataClass[l][f] + k) / (0.0 + self.dataClass[l]["occur"] + 2*k)

      #Occurence of k    
      freq = 0
      #Valid data size 
      vData = len(validationData)
      #Loop for all valid data in data set
      for i in range(vData):
        #Calculate log of probability using given formula
        probLog = self.calculateLogJointProbabilities(validationData[i]).argMax()
        if(probLog == validationLabels[i]):
          freq += 1
      #Check for optimum k    
      if freq > oFreq:
        optimumK = k
        #Update the value
        oFreq = freq     
      self.k = optimumK
      #Get legal labels
      labels = self.legalLabels
      #Loop for all valid labels
      for l in labels:
        #Get features
        features = self.features
        #Loop for every feature 
        for f in features:
          self.classProb[l][f] = (self.dataClass[l][f] + self.k) / (0.0 + self.dataClass[l]["occur"] + 2*self.k)
    
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for featureCounter in testData:
      posterior = self.calculateLogJointProbabilities(featureCounter)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, featureCounter):
    """
    Returns the log-joint distribution over legal labels and the featureCounter.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, featureCounter) )>
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    #Get valid labels
    labels = self.legalLabels
    #Loop for all labels
    for l in labels:
      #Calculate log of probability using math.log
      logProb = math.log(self.dataClass[l]["occur"] / (self.dataSize + 0.0))
      #Get features
      features = self.features
      #Loop for all features
      for f in features:
        prob = self.classProb[l][f]
        #Compare datum or feature counter
        if(featureCounter[f]):
          logProb += math.log(prob)
        else:
          logProb += math.log(1 - prob)
      logJoint[l] = logProb 
    #Return log-probability  
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    """
    oddFeatures = []
    oddFeaturesCounter = util.Counter()

    #Get features 
    features = self.features
    #Loop for all features
    for f in features:
      #Calculate prob of lable1 for each pixel feature
      probL1 = self.classProb[label1][f]
      #Calculate prob of lable2 for each pixel feature
      probL2 = self.classProb[label2][f]
      #Find odd ratio using formula
      oddFeaturesCounter[f] = probL1/probL2
    #Sort odd-features  
    oddFeatures = oddFeaturesCounter.sortedKeys()
    #Return list of 100 highest odds
    return oddFeatures[:100]
    
