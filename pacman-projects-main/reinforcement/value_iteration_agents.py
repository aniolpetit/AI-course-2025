# value_iteration_agents.py
# -----------------------
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


# value_iteration_agents.py
# -----------------------
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


import mdp, util

from learning_agents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state, action, next_state)
              mdp.is_terminal(state)
        """
        super().__init__()
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.run_value_iteration()
    
    def run_value_iteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        # Perform the batch value iteration update for the requested number of iterations.
        for _ in range(self.iterations):
            old_values = self.values.copy()
            new_values = util.Counter()

            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    # Terminal states do not collect more reward, so their value stays at zero.
                    new_values[state] = 0
                    continue

                actions = self.mdp.get_possible_actions(state)
                if not actions:
                    new_values[state] = 0
                    continue

                # Evaluate every action using V_k (old_values) and keep the max.
                best_value = float("-inf")
                for action in actions:
                    q_value = 0
                    for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
                        reward = self.mdp.get_reward(state, action, next_state)
                        q_value += prob * (reward + self.discount * old_values[next_state])
                    best_value = max(best_value, q_value)

                new_values[state] = best_value

            self.values = new_values
            
    def get_value(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0
        for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            # Add the discounted value of the successor state weighted by transition probability.
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.get_possible_actions(state)
        if not actions:
            # Terminal states (or states with no legal actions) return no action.
            return None

        # Track the action with the largest Q-value under the current value function.
        best_action, best_value = None, float("-inf")
        for action in actions:
            q_value = self.compute_q_value_from_values(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action

    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        return self.compute_q_value_from_values(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learning_agents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state)
              mdp.is_terminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        states = self.mdp.get_states()
        if not states:
            return

        for iteration in range(self.iterations):
            state = states[iteration % len(states)]
            if self.mdp.is_terminal(state):
                # Terminal states never change, so skip the update.
                continue

            actions = self.mdp.get_possible_actions(state)
            if not actions:
                continue

            # Update only the selected state using the most recent values.
            best_value = float("-inf")
            for action in actions:
                q_value = 0
                for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
                    reward = self.mdp.get_reward(state, action, next_state)
                    q_value += prob * (reward + self.discount * self.values[next_state])
                best_value = max(best_value, q_value)

            self.values[state] = best_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """*** YOUR CODE HERE ***"""

