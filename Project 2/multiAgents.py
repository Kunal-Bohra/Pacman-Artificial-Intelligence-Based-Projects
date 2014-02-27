from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    oldFoodList = currentGameState.getFood().asList()
    newFoodList = successorGameState.getFood().asList()
    pacPos = tuple(successorGameState.getPacmanPosition())
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #Calculate ghost-postion from current state of pacman.  
    for ghostState in newGhostStates:
      #Assign ghost-postion.
      ghostPos = ghostState.getPosition()
      #Check if ghost and pacman collide.
      if ghostPos == pacPos:
            return -100
          
    #Calculate manhattan distance of food from pacman.
    for food in oldFoodList:
      #Use manhattan distance formula.
      man_dis = [manhattanDistance(food,pacPos)]
      #Sort the man_dis list.
      man_dis.sort()
      #Take the first value from list.
      value = man_dis[0]
      if Directions.STOP in action:  
        return -100
    return (-value) 

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)



class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    #The max function take a gameState, depth of tree and agentIndex.
    #It comoutes the maximum value in Minimax algorithm for max-palyer.

    def max_fun(self, gameState, depth, agentIndex):
      #The index for pacman is 0.
      pacmanIndex = 0
      #pacman_actions denotes the legal actions that it can take.
      pacman_actions = gameState.getLegalActions(pacmanIndex)
      #ubound denotes the negative infinity or high value for Minimax algorithm.
      ubound = -100000
      #Terminal test to Check if we have reach the cut-off state or leaf node.
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      #Loop to generate successors.
      for action in pacman_actions:
        #Removing Directions.STOP from legal actions as given in question.
        if action != Directions.STOP:
          #Generate successor for the pacman using action from actions.
          next_node = gameState.generateSuccessor(pacmanIndex, action)
          #Minimize next agent
          ghostIndex = pacmanIndex+1
          value = self.min_fun(next_node, depth, ghostIndex)
          #Check if value is greater than negative infinity.  
          if value > ubound: # and action!= Directions.STOP:
            #Update value of negative infinity
            ubound = max(value,ubound)
            #Update the action taken by max-player.
            max_result = action
        #Return ation taken for depth being 1.
      return(ubound, max_result) [depth ==1]


    #The min_fun take a gameState, depth of tree and agentIndex.
    #It computes the minimum value in MinMax algorith for min-player.

    def min_fun(self, gameState, depth, agentIndex):
      #Ghost actions denotes legal action the ghost agent can take.
      ghost_actions = gameState.getLegalActions(agentIndex)
      #lbound denotes the positive inifinity value of MinMax algorithm.
      lbound = 100000
      #agent_count dentoes the total number of enemy agents in game.
      agent_count = gameState.getNumAgents()
      #Terminal test to check if the state is terminal state so as to cut-off.
      if gameState.isLose():
        return self.evaluationFunction(gameState)
      #Loop for every action in legal ghost/agent actions.
      for action in ghost_actions:
        #Remove action from legal actions according to question.
        if action!= Directions.STOP:
          next_node = gameState.generateSuccessor(agentIndex, action)
          #Decrement the agent_count to check if ghost/agent left.
          if agentIndex == agent_count-1 :
            #Check if leaf node reached.
            if depth == self.depth:
              value = self.evaluationFunction(next_node)
            #Else call max_fun to maximize value in next ply.
            else:
              #Maximize for pacman.
              pacmanIndex=0
              value = self.max_fun(next_node,depth+1,pacmanIndex)
          else:
            #For remaining ghosts, minimize the value.
            value = self.min_fun(next_node, depth, agentIndex+1)
        #Update the value of positive infinity
        if value < lbound: # and action!= Directions.STOP:
          lbound = min(value,lbound)
          min_result = action
      return lbound

    #The minmax function computes the action taken to maximize the value for max player.
    def minmax(self, gameState):
      depth = 0
      depth += 1
      pacmanIndex = 0
      max_result = self.max_fun(gameState, depth, pacmanIndex)
      return max_result


    def getAction(self, gameState):

      """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
      return self.minmax(gameState)

      #util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    #The max function take a gameState, depth of tree and agentIndex.
    #It comoutes the maximum value in AlphaBeta pruning algorithm for max-palyer.

    def max_fun(self, gameState, depth, agentIndex, alpha, beta):
      #The index for pacman is 0.
      pacmanIndex = 0
      #pacman_actions denotes the legal actions that it can take.
      pacman_actions = gameState.getLegalActions(pacmanIndex)
      #ubound denotes the negative infinity or high value for Minimax algorithm.
      ubound = -100000
      #Terminal test to Check if we have reach the cut-off state or leaf node.
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      #Loop to generate successors.
      for action in pacman_actions:
        #Removing Directions.STOP from legal actions as given in question.
        if action != Directions.STOP:
          #Generate successor for the pacman using action from actions.
          next_node = gameState.generateSuccessor(pacmanIndex, action)
          #Minimize next agent.
          ghostIndex = pacmanIndex+1
          value = self.min_fun(next_node, depth,ghostIndex, alpha, beta)
          if value > beta:
            #Update value to remove the unvisited branch of tree.
            return value
          #Check if value is greater than negative infinity.
          if value > ubound: # and action!= Directions.STOP:
            #Update value of negative infinity
            ubound = max(value,ubound)
            #Update the action taken by max-player.
            max_result = action
          #Update alpha as per algorithm
          alpha = max(alpha, ubound)
        #Return ation taken for depth being 1.
        #Else return the new value of negative infinity
      return(ubound, max_result) [depth ==1]


    #The min_fun take a gameState, depth of tree and agentIndex.
    #It computes the minimum value in AlphaBeta algorithm for min-player.

    def min_fun(self, gameState, depth, agentIndex, alpha, beta):
      #Ghost actions denotes legal action the ghost agent can take.
      ghost_actions = gameState.getLegalActions(agentIndex)
      #lbound denotes the positive inifinity value of MinMax algorithm.
      lbound = 100000
      #agent_count dentoes the total number of enemy agents in game.
      agent_count = gameState.getNumAgents()
      #Terminal test to check if the state is terminal state so as to cut-off.
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      #Loop for every action in legal ghost/agent actions.
      for action in ghost_actions:
        #Remove action from legal actions according to question.
        if action!= Directions.STOP:
          next_node = gameState.generateSuccessor(agentIndex, action)
          #Decrement the agent_count to check if ghost/agent left.
          if agentIndex == agent_count-1 :
            #Check if leaf node reached.
            if depth == self.depth:
              value = self.evaluationFunction(next_node)
            #Else call max_fun to maximize value in next ply.
            else:
              pacmanIndex = 0
              #Maximize for pacman.
              value = self.max_fun(next_node,depth+1,pacmanIndex, alpha, beta)
          else:
            #For remaining ghosts, minimize the value.
            value = self.min_fun(next_node, depth, agentIndex+1, alpha, beta)
        #Update value to remove the unvisited branch of tree.
        if value < alpha:
          return value
        #Update the value of positive infinity
        if value < lbound: # and action!= Directions.STOP:
          lbound = min(value,lbound)
          min_result = action
        #Update beta as per algorithm
        beta = min(beta, value)
      return lbound

    #The minmax function computes the action taken to maximize the value for max player.
    def minmax(self, gameState):
      depth = 0
      depth += 1
      pacmanIndex = 0
      max_result = self.max_fun(gameState, depth, pacmanIndex, -100000, 100000)
      return max_result


    def getAction(self, gameState):

      """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
      return self.minmax(gameState)

      #util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    return self.expectimax(gameState)
    util.raiseNotDefined()


    #The max function take a gameState, depth of tree and agentIndex.
    #It comoutes the maximum value in Minimax algorithm for max-palyer.

  def max_fun(self, gameState, depth, agentIndex):
    #The index for pacman is 0.
    pacmanIndex = 0
    #pacman_actions denotes the legal actions that it can take.
    pacman_actions = gameState.getLegalActions(pacmanIndex)
    #ubound denotes the negative infinity or high value for Minimax algorithm.
    ubound = -100000
    #Terminal test to Check if we have reach the cut-off state or leaf node.
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)
    #Loop to generate successors.
    for action in pacman_actions:
      #Removing Directions.STOP from legal actions as given in question.
      if action != Directions.STOP:
        #Generate successor for the pacman using action from actions.
        next_node = gameState.generateSuccessor(pacmanIndex, action)
        #Minimize ghost-agent.
        ghostIndex = pacmanIndex+1
        value = self.min_fun(next_node, depth, ghostIndex)
        #Check if value is greater than negative infinity.
        if value > ubound: # and action!= Directions.STOP:
          #Update value of negative infinity
          ubound = max(value,ubound)
          #Update the action taken by max-player.
          max_result = action
      #Return ation taken for depth being 1.
      #Else return the new value of negative infinity
    return(ubound, max_result) [depth ==1]
   


    #The min_fun take a gameState, depth of tree and agentIndex.
    #It computes the minimum value in MinMax algorith for min-player.

  def min_fun(self, gameState, depth, agentIndex):
    #Ghost actions denotes legal action the ghost agent can take.
    ghost_actions = gameState.getLegalActions(agentIndex)
    #lbound denotes the positive inifinity value of MinMax algorithm.
    lbound = 0
    #agent_count dentoes the total number of enemy agents in game.
    agent_count = gameState.getNumAgents()
    #Terminal test to check if the state is terminal state so as to cut-off.
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)
    #Calculating expected value for next ply.
    #expected_value = 1.0 / (1+len(ghost_actions))
    expected_value = 1.0 / len(ghost_actions)
    #Loop for every action in legal ghost/agent actions.
    for action in ghost_actions:
      #Remove action from legal actions according to question.
      if action!= Directions.STOP:
        next_node = gameState.generateSuccessor(agentIndex, action)
        #Decrement the agent_count to check if ghost/agent left.
        if agentIndex == agent_count-1 :
          #Check if leaf node reached.
          if depth == self.depth:
            value = self.evaluationFunction(next_node)
          #Else call max_fun to maximize value in next ply.
          else:
            #Maximize for pacman.
            pacmanIndex = 0
            value = self.max_fun(next_node,depth+1,pacmanIndex)
        else:
          #For remaining ghosts, minimize the value.
          value = self.min_fun(next_node, depth, agentIndex+1)
        effective_value=value*expected_value
        lbound=lbound+effective_value
    return lbound


  def expectimax(self, gameState):
      depth = 0
      depth += 1
      pacmanIndex = 0
      max_result = self.max_fun(gameState, depth, pacmanIndex)
      return max_result



def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Our evaluation function uses distance between ghosts and pacman as well as distance between pacman and capsules
                 along with current game score. The distance between ghost and pacman is subtracted from current score and distance
                 between capsules and pacman is added to current score. 
  """
  "*** YOUR CODE HERE ***"
    
  capsules = currentGameState.getCapsules()   
  newFoodList = currentGameState.getFood().asList()
  ghostStates = currentGameState.getGhostStates()
  pacman_pos = currentGameState.getPacmanPosition()
  current_score = currentGameState.getScore()
  ghost_score = 0
  cap_score = 0

  #Calculate the distance between pacman and capsules in game using Manhattan distance.
  #Check if capsule list is not empty.
  if(len(capsules) != 0):
    #Use manhattan distance formula
    for capsule in capsules:
      cap_dis = min([manhattanDistance(capsule, pacman_pos)])
      if cap_dis == 0 :
        cap_score = float(1)/cap_dis
      else:
        cap_score = -100
        
  #Calculate distance between ghosts and pacman using Manhattan distance.s        
  for ghost in ghostStates:
    ghost_x = (ghost.getPosition()[0])
    ghost_y = (ghost.getPosition()[1])
    ghost_pos = ghost_x,ghost_y
    ghost_dis = manhattanDistance(pacman_pos, ghost_pos)

  #Evaluation function returms following scores.
  return current_score  - (1.0/1+ghost_dis)  + (1.0/1+cap_score)


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"




