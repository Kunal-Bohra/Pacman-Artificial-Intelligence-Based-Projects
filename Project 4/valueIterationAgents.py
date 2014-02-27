# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        #Iteration loop
        for i in range(0,self.iterations):
            #get old value for every iteration
            oldVal = self.values.copy()
            #Calculate Qvalue for every state
            for state in self.mdp.getStates():
                #Initialize a Qvalue tuple
                qValue = []
                actions = self.mdp.getPossibleActions(state)
                #For every action, do
                for action in actions: 
                    #Calculate Qvalue for this state
                    newqVal = 0
                    #Get transistion probability from mdp.py
                    p = mdp.getTransitionStatesAndProbs(state, action)
                    for nextState, probability in p:
                        #Calculate the reward using mdp.py
                        reward = self.mdp.getReward(state, action, nextState)
                        #Formula from text
                        newqVal += probability*(reward + (self.discount*oldVal[nextState]))
                    #Append the tuple initialized above with a new QValue  
                    qValue.append(newqVal)
                if len(qValue) > 0:
                    var = max(qValue)
                    self.values[state] = var 
                else:
                    pass

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def getPolicy(self, state):
        #Return None for Terminal state
        if self.mdp.isTerminal(state):
            return None
        #Inittialize max value at beginning to infinity
        qMax = -10000
        #Initialize max action to None at start
        actionMax = None
        #Get actions from mdp.py
        actions = self.mdp.getPossibleActions(state)
        #Loop for every possible acrion to make policy
        for probableAction in actions:
            #Calculate new QValue
            newQVal = self.getQValue(state, probableAction)
            #We don't use argmax here
            if newQVal > qMax:
                qMax = newQVal
                actionMax = probableAction
        #Return max action        
        return actionMax

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        
        #Return None for Terminal state
        if self.mdp.isTerminal(state):
            return None
        #Inittialize max value at beginning to infinity
        qMax = -10000
        #Initialize max action to None at start
        actionMax = None
        #Get actions from mdp.py
        actions = self.mdp.getPossibleActions(state)
        #Loop for every possible acrion to make policy
        for probableAction in actions:
            #Calculate new QValue
            newQVal = self.getQValue(state, probableAction)
            #We don't use argmax here
            if newQVal > qMax:
                qMax = newQVal
                actionMax = probableAction
        #Return max action        
        return actionMax

    def getQValue(self, state, action):
      #Initialize QValue to 0 at start  
      qValue = 0
      #Get transistion probability from mdp.py
      probability = self.mdp.getTransitionStatesAndProbs(state, action)
      #Loop for calculating qvalue
      for nextState, prob in probability:
        #Get reward from mdp.py
        reward = self.mdp.getReward(state, action, nextState)
        #Formula from text
        qValue  += prob * (reward + (self.discount * self.getValue(nextState)))
      return qValue
