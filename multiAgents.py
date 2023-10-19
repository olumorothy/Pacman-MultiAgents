# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"
        score = 0
        ghostPosition = newGhostStates[0].configuration.pos
        closetGhost = manhattanDistance(newPos, ghostPosition)

        newFoodPositions = newFood.asList()
        foodDistances = [manhattanDistance(newPos, foodPositions) for foodPositions in newFoodPositions]

        if len(foodDistances)==0:
            return 0

        closestFood = min(foodDistances)

        if action == 'Stop':
            score-=50

        return successorGameState.getScore() + closetGhost /(closestFood * 10) + score
        
        util.raiseNotDefined()
        return successorGameState.getScore()

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        result = self.getValue(gameState,0,0)
        return result[1]
        util.raiseNotDefined()
    
    def getValue(self, gameState,index,depth):
        #Return a tuple of [score,action]
        #terminal state
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(),""

        #When pacman has index of 0, i.e max agent
        if index == 0:
            return self.max_function(gameState, index, depth)
        else:
            #min agent
            return self.min_function(gameState, index, depth)
    
    def max_function(self, gameState, index, depth):
        #Returns the maximum value-action for the max agent
        possibleMoves = gameState.getLegalActions(index)
        maxValue = float("-inf")
        maxAction = ""

        for action in possibleMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            #updating the succesor agent index and depth
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth +=1
            current_value = self.getValue(successor, successor_index,successor_depth)[0]

            if current_value > maxValue:
                maxValue = current_value
                maxAction = action
        return maxValue, maxAction

    def min_function(self, gameState, index, depth):
        #Min utility value action for min agent
        possibleMoves = gameState.getLegalActions(index)
        
        minValue = float('inf')
        minAction = ""

        for action in possibleMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            #update agent's index
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth +=1
            
            current_value = self.getValue(successor, successor_index, successor_depth)[0]

            if current_value < minValue:
                minValue = current_value
                minAction = action
        return minValue, minAction


            



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        result = self.getBestActionAndScore(gameState, 0, 0, float("-inf"),float("inf"))

        return result[0]
        util.raiseNotDefined()
    def getBestActionAndScore(self,gameState,index, depth, alpha, beta):
        #Returns a tuple of action and state

        #checking for terminal state
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "",gameState.getScore()
        
        #for max-agent, pacman has index of 0
        if index == 0:
            return self.maxValue(gameState,index,depth,alpha,beta)
        else:
            return self.minValue(gameState,index,depth,alpha,beta)
    
    def maxValue(self,gameState, index, depth,alpha,beta):

        possibleMoves = gameState.getLegalActions(index)
        maxValue = float("-inf")
        maxAction= ""

        for action in possibleMoves:
            successor = gameState.generateSuccessor(index,action)
            successor_index = index + 1
            successor_depth = depth

            #update successor agents
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            #Calculate the action-score for the current successor
            
            currentAction, currentValue \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            #updating maxvalue and maxaction for the max agent
            if currentValue > maxValue:
                maxValue = currentValue
                maxAction = action
            
            #update value of alpha for the maximizer
            alpha = max(alpha, maxValue)

            if maxValue > beta:
                return maxAction,maxValue
        return maxAction, maxValue
    
    def minValue(self,gameState, index, depth, alpha, beta):
        #returns utility action,score for min-agent without alpha-beta prunning

        possibleMoves = gameState.getLegalActions(index)
        minValue = float("inf")
        minAction = ""

        for action in possibleMoves:
            successor = gameState.generateSuccessor(index,action)
            successor_index = index + 1
            successor_depth = depth

            #Update the successor agent index nd depth

            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            #action score for the successor
            currentAction, currentValue \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

                #Update the min and max for minimizer
            if currentValue < minValue:
                minValue = currentValue
                minAction = action

            #update beta for minimizer
            beta = min(beta,minValue)

            if minValue < alpha:
                return minAction,minValue

        return minAction,minValue

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
        "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** TTU CS 5368 Fall 2023 YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
