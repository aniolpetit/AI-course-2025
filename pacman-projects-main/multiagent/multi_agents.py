# multi_agents.py
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


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
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


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"
        
        # Start with the base score of the successor state
        # This accounts for eating food, dying, winning, and time penalty
        score = successor_game_state.get_score()
        
        # FOOD EVALUATION:
        # Get list of all food positions from the grid
        food_list = new_food.as_list()
        
        # Calculate distance to closest food pellet
        # We want Pacman to be attracted to the nearest food
        if food_list:
            # Compute Manhattan distance to each food pellet
            food_distances = [manhattan_distance(new_pos, food) for food in food_list]
            min_food_distance = min(food_distances)
            
            # Use the reciprocal of distance to reward being close to food
            # Adding 1 to denominator to avoid division by zero if Pacman is on food
            # Multiply by 10 to make this feature more influential
            score += 10.0 / (min_food_distance + 1)
        
        # GHOST EVALUATION:
        # We need to consider both dangerous ghosts and scared ghosts differently
        for ghost_state, scared_time in zip(new_ghost_states, new_scared_times):
            ghost_pos = ghost_state.get_position()
            distance_to_ghost = manhattan_distance(new_pos, ghost_pos)
            
            if scared_time > 0:
                # Ghost is scared - we should chase it for points!
                # Reward being close to scared ghosts
                # Only chase if we have enough time to reach it
                if distance_to_ghost < scared_time:
                    score += 100.0 / (distance_to_ghost + 1)
            else:
                # Ghost is dangerous - we should avoid it!
                # Heavily penalize being too close to active ghosts
                if distance_to_ghost <= 1:
                    # Very dangerous - almost certain death
                    score -= 500
                elif distance_to_ghost <= 2:
                    # Still dangerous - strong penalty
                    score -= 100
                else:
                    # Far enough to be safe, small penalty to encourage distance
                    score -= 10.0 / (distance_to_ghost + 1)
        
        # STOPPING PENALTY:
        # Discourage Pacman from stopping unless necessary
        # This keeps the agent active and prevents it from standing still
        if action == Directions.STOP:
            score -= 10
        
        return score

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

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

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        # Start minimax search from Pacman's perspective (agent 0) at depth 0
        # We want to return the action, not the value, so we track the best action
        _, best_action = self.minimax(game_state, 0, 0)
        return best_action
    
    def minimax(self, game_state, agent_index, current_depth):
        """
        Recursive minimax algorithm that returns (value, action) tuple.
        
        Parameters:
        - game_state: Current state of the game
        - agent_index: Index of the current agent (0 for Pacman, 1+ for ghosts)
        - current_depth: Current depth in the search tree
        
        Returns:
        - (value, action): Best value and corresponding action for the current agent
        
        Algorithm explanation:
        - Pacman (agent 0) is the MAX player, tries to maximize the score
        - Ghosts (agents 1+) are MIN players, try to minimize the score
        - A full ply consists of all agents (Pacman + all ghosts) making one move
        - We increment depth only after all agents have moved (when we return to Pacman)
        """
        
        # TERMINAL CONDITIONS:
        # Check if we've reached a terminal state or maximum depth
        if game_state.is_win() or game_state.is_lose() or current_depth == self.depth:
            # Return the evaluation of this state and no action (terminal node)
            return self.evaluation_function(game_state), None
        
        # DETERMINE NEXT AGENT:
        # Calculate the next agent's index
        # After the last ghost, we return to Pacman (agent 0) and increment depth
        num_agents = game_state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents
        
        # Increment depth counter when we complete a full ply (after last ghost moves)
        next_depth = current_depth + 1 if next_agent == 0 else current_depth
        
        # Get legal actions for the current agent
        legal_actions = game_state.get_legal_actions(agent_index)
        
        # Handle case where agent has no legal actions (shouldn't happen in practice)
        if not legal_actions:
            return self.evaluation_function(game_state), None
        
        # MINIMAX LOGIC:
        if agent_index == 0:
            # MAX NODE (Pacman's turn)
            # Pacman tries to maximize the score
            return self.max_value(game_state, agent_index, current_depth, next_agent, next_depth, legal_actions)
        else:
            # MIN NODE (Ghost's turn)
            # Ghosts try to minimize the score
            return self.min_value(game_state, agent_index, current_depth, next_agent, next_depth, legal_actions)
    
    def max_value(self, game_state, agent_index, current_depth, next_agent, next_depth, legal_actions):
        """
        Implements the MAX part of minimax for Pacman.
        Pacman tries to maximize the score by choosing the action that leads to the highest value.
        
        Returns:
        - (max_value, best_action): The maximum value and the action that achieves it
        """
        max_val = float('-inf')  # Initialize to negative infinity
        best_action = None
        
        # Evaluate each possible action
        for action in legal_actions:
            # Generate the successor state after taking this action
            successor_state = game_state.generate_successor(agent_index, action)
            
            # Recursively get the value of this successor state
            # We only care about the value, not the action from deeper levels
            value, _ = self.minimax(successor_state, next_agent, next_depth)
            
            # Update max value and best action if this action is better
            if value > max_val:
                max_val = value
                best_action = action
        
        return max_val, best_action
    
    def min_value(self, game_state, agent_index, current_depth, next_agent, next_depth, legal_actions):
        """
        Implements the MIN part of minimax for ghosts.
        Ghosts try to minimize the score by choosing the action that leads to the lowest value.
        
        Returns:
        - (min_value, best_action): The minimum value and the action that achieves it
        """
        min_val = float('inf')  # Initialize to positive infinity
        best_action = None
        
        # Evaluate each possible action
        for action in legal_actions:
            # Generate the successor state after taking this action
            successor_state = game_state.generate_successor(agent_index, action)
            
            # Recursively get the value of this successor state
            value, _ = self.minimax(successor_state, next_agent, next_depth)
            
            # Update min value and best action if this action is better (lower)
            if value < min_val:
                min_val = value
                best_action = action
        
        return min_val, best_action
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        
        # Start alpha-beta search from Pacman's perspective (agent 0) at depth 0
        # Initialize alpha to -infinity (best MAX can guarantee so far)
        # Initialize beta to +infinity (best MIN can guarantee so far)
        _, best_action = self.alpha_beta(game_state, 0, 0, float('-inf'), float('inf'))
        return best_action
    
    def alpha_beta(self, game_state, agent_index, current_depth, alpha, beta):
        """
        Recursive alpha-beta pruning algorithm that returns (value, action) tuple.
        
        Parameters:
        - game_state: Current state of the game
        - agent_index: Index of the current agent (0 for Pacman, 1+ for ghosts)
        - current_depth: Current depth in the search tree
        - alpha: Best value MAX can guarantee (lower bound for MAX)
        - beta: Best value MIN can guarantee (upper bound for MIN)
        
        Returns:
        - (value, action): Best value and corresponding action for the current agent
        
        Alpha-Beta Pruning explanation:
        - Alpha represents the best value that the MAX player (Pacman) can guarantee
          along the path to the current node. It's initialized to -infinity.
        - Beta represents the best value that the MIN player (ghosts) can guarantee
          along the path to the current node. It's initialized to +infinity.
        - If at any point we find that beta <= alpha, we can prune (stop exploring)
          because the current branch cannot influence the final decision.
        - IMPORTANT: We only prune on strict inequality (beta <= alpha), not equality,
          to match the autograder's expectations.
        """
        
        # TERMINAL CONDITIONS:
        # Check if we've reached a terminal state or maximum depth
        if game_state.is_win() or game_state.is_lose() or current_depth == self.depth:
            # Return the evaluation of this state and no action (terminal node)
            return self.evaluation_function(game_state), None
        
        # DETERMINE NEXT AGENT:
        # Calculate the next agent's index
        # After the last ghost, we return to Pacman (agent 0) and increment depth
        num_agents = game_state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents
        
        # Increment depth counter when we complete a full ply (after last ghost moves)
        next_depth = current_depth + 1 if next_agent == 0 else current_depth
        
        # Get legal actions for the current agent
        legal_actions = game_state.get_legal_actions(agent_index)
        
        # Handle case where agent has no legal actions (shouldn't happen in practice)
        if not legal_actions:
            return self.evaluation_function(game_state), None
        
        # ALPHA-BETA LOGIC:
        if agent_index == 0:
            # MAX NODE (Pacman's turn)
            # Pacman tries to maximize the score while updating alpha
            return self.max_value_alpha_beta(game_state, agent_index, current_depth, 
                                            next_agent, next_depth, legal_actions, alpha, beta)
        else:
            # MIN NODE (Ghost's turn)
            # Ghosts try to minimize the score while updating beta
            return self.min_value_alpha_beta(game_state, agent_index, current_depth, 
                                            next_agent, next_depth, legal_actions, alpha, beta)
    
    def max_value_alpha_beta(self, game_state, agent_index, current_depth, 
                             next_agent, next_depth, legal_actions, alpha, beta):
        """
        Implements the MAX part of alpha-beta pruning for Pacman.
        Pacman tries to maximize the score and updates alpha.
        Prunes branches when a value >= beta is found (MIN won't let us get here).
        
        Parameters:
        - alpha: Best value MAX can guarantee on the path to root
        - beta: Best value MIN can guarantee on the path to root
        
        Returns:
        - (max_value, best_action): The maximum value and the action that achieves it
        """
        max_val = float('-inf')  # Initialize to negative infinity
        best_action = None
        
        # Evaluate each possible action in order (important for autograder)
        for action in legal_actions:
            # Generate the successor state after taking this action
            successor_state = game_state.generate_successor(agent_index, action)
            
            # Recursively get the value of this successor state
            # Pass alpha and beta down for pruning decisions
            value, _ = self.alpha_beta(successor_state, next_agent, next_depth, alpha, beta)
            
            # Update max value and best action if this action is better
            if value > max_val:
                max_val = value
                best_action = action
            
            # PRUNING CONDITION for MAX node:
            # If the value we found is greater than beta, the MIN player above
            # will never choose this path (they already have a better option).
            # We can stop exploring siblings.
            # NOTE: We only prune on strict inequality (value > beta), not equality
            if max_val > beta:
                return max_val, best_action
            
            # Update alpha to be the maximum of current alpha and the value found
            # Alpha represents the best value MAX can guarantee
            alpha = max(alpha, max_val)
        
        return max_val, best_action
    
    def min_value_alpha_beta(self, game_state, agent_index, current_depth, 
                             next_agent, next_depth, legal_actions, alpha, beta):
        """
        Implements the MIN part of alpha-beta pruning for ghosts.
        Ghosts try to minimize the score and update beta.
        Prunes branches when a value <= alpha is found (MAX won't let us get here).
        
        Parameters:
        - alpha: Best value MAX can guarantee on the path to root
        - beta: Best value MIN can guarantee on the path to root
        
        Returns:
        - (min_value, best_action): The minimum value and the action that achieves it
        """
        min_val = float('inf')  # Initialize to positive infinity
        best_action = None
        
        # Evaluate each possible action in order (important for autograder)
        for action in legal_actions:
            # Generate the successor state after taking this action
            successor_state = game_state.generate_successor(agent_index, action)
            
            # Recursively get the value of this successor state
            # Pass alpha and beta down for pruning decisions
            value, _ = self.alpha_beta(successor_state, next_agent, next_depth, alpha, beta)
            
            # Update min value and best action if this action is better (lower)
            if value < min_val:
                min_val = value
                best_action = action
            
            # PRUNING CONDITION for MIN node:
            # If the value we found is less than alpha, the MAX player above
            # will never choose this path (they already have a better option).
            # We can stop exploring siblings.
            # NOTE: We only prune on strict inequality (value < alpha), not equality
            if min_val < alpha:
                return min_val, best_action
            
            # Update beta to be the minimum of current beta and the value found
            # Beta represents the best value MIN can guarantee
            beta = min(beta, min_val)
        
        return min_val, best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
