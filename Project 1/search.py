# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
from game import Directions

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
         
class Dir_tuple():
    
    def __init__(self):
        pass
    action = {'North': Directions.NORTH,'East': Directions.EAST, 'West': Directions.WEST, 'South': Directions.SOUTH}
            
def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]
  
  
def algo_search(problem,openList):
  dirTupleObj=Dir_tuple()
  
  start_state=problem.getStartState()
  start_node=[start_state,[],0]
  openList.push(start_node)
  closeList=[] 
   
  
  while not openList.isEmpty():
    if (openList.isEmpty()):
        break
    current_node=openList.pop()      
    if (problem.isGoalState(current_node[0])):
        return current_node[1]
    if current_node[0] in closeList:
      pass
    else:
      child_list=problem.getSuccessors(current_node[0])
      for child in child_list:
        openList.push((child[0], current_node[1] + [dirTupleObj.action[child[1]]], child[2]))
      closeList += [current_node[0]]
  return []    

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  openList=util.Stack()
  return algo_search(problem,openList)

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  openList=util.Queue() 
  return algo_search(problem,openList)
 
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  def priority(current_node):
        return current_node[2]
  
  openList=util.PriorityQueueWithFunction(priority) 
  return algo_search(problem,openList)

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  def priority(current_node):
        cost=current_node[2]+heuristic(current_node[0],problem)
        return cost
  
  openList=util.PriorityQueueWithFunction(priority) 
  return algo_search(problem,openList)
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
