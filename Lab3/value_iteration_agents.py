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
        "*** YOUR CODE HERE ***"
        # fem tantes iteracions com ens demanen a self.iterations
        for i in range(self.iterations):
            # creem un Counter temporal per fer el batch update i no trepitjar els valors de la mateixa iteracio
            nous_valors = util.Counter()

            # recorrem tots els estats del MDP
            for state in self.mdp.get_states():
                # si l'estat es terminal no te accions possibles, el deixem a 0
                if self.mdp.is_terminal(state):
                    continue

                # calculem el maxim Q-value de les accions legals per actualitzar l'estat
                accions = self.mdp.get_possible_actions(state)
                if len(accions) > 0:
                    max_q = float("-inf")
                    for action in accions:
                        q_val = self.compute_q_value_from_values(state, action)
                        if q_val > max_q:
                            max_q = q_val
                    nous_valors[state] = max_q

            # actualitzem tots els valors de cop al final de la iteracio
            self.values = nous_valors
            
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
        "*** YOUR CODE HERE ***"
        # calculem el Q-value sumant per cada estat seguent: prob * (reward + discount * valor_estat_seguent)
        q_value = 0.0
        transicions = self.mdp.get_transition_states_and_probs(state, action)

        for next_state, prob in transicions:
            recompensa = self.mdp.get_reward(state, action, next_state)
            q_value += prob * (recompensa + self.discount * self.values[next_state])

        return q_value

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # si estem en un estat terminal no hi ha cap accio a fer
        if self.mdp.is_terminal(state):
            return None

        accions = self.mdp.get_possible_actions(state)
        if len(accions) == 0:
            return None

        # busquem quina accio ens dona el Q-value mes alt
        millor_accio = None
        max_q = float("-inf")

        for action in accions:
            q_val = self.compute_q_value_from_values(state, action)
            if q_val > max_q:
                max_q = q_val
                millor_accio = action

        return millor_accio

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
        """*** YOUR CODE HERE ***"""
        # obtenim la llista d'estats per anar actualitzant un per un de forma ciclica
        estats = self.mdp.get_states()
        num_estats = len(estats)

        for i in range(self.iterations):
            state = estats[i % num_estats]

            # si es terminal no fem res en aquesta iteracio
            if self.mdp.is_terminal(state):
                continue

            accions = self.mdp.get_possible_actions(state)
            if len(accions) > 0:
                max_q = float("-inf")
                for action in accions:
                    q_val = self.compute_q_value_from_values(state, action)
                    if q_val > max_q:
                        max_q = q_val
                self.values[state] = max_q

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
        # 1. calculem els predecessors de tots els estats
        predecessors = {}
        estats = self.mdp.get_states()
        for s in estats:
            predecessors[s] = set()

        for s in estats:
            if not self.mdp.is_terminal(s):
                for action in self.mdp.get_possible_actions(s):
                    for next_state, prob in self.mdp.get_transition_states_and_probs(s, action):
                        if prob > 0:
                            predecessors[next_state].add(s)

        # 2. inicialitzem la cua de prioritats buida
        cua = util.PriorityQueue()

        # 3. per cada estat no terminal calculem la diferencia (error de Bellman) i l'afegim a la cua
        for s in estats:
            if not self.mdp.is_terminal(s):
                accions = self.mdp.get_possible_actions(s)
                if len(accions) > 0:
                    max_q = float("-inf")
                    for action in accions:
                        q_val = self.compute_q_value_from_values(s, action)
                        if q_val > max_q:
                            max_q = q_val
                    diff = abs(self.values[s] - max_q)
                    cua.push(s, -diff)

        # 4. fem el bucle d'iteracions actualitzant els estats amb mes prioritat
        for i in range(self.iterations):
            if cua.is_empty():
                break

            s = cua.pop()
            if not self.mdp.is_terminal(s):
                accions = self.mdp.get_possible_actions(s)
                if len(accions) > 0:
                    max_q = float("-inf")
                    for action in accions:
                        q_val = self.compute_q_value_from_values(s, action)
                        if q_val > max_q:
                            max_q = q_val
                    self.values[s] = max_q

            # revisem els predecessors de s per si cal actualitzar la seva prioritat
            for p in predecessors[s]:
                if not self.mdp.is_terminal(p):
                    accions_p = self.mdp.get_possible_actions(p)
                    if len(accions_p) > 0:
                        max_q_p = float("-inf")
                        for action in accions_p:
                            q_val_p = self.compute_q_value_from_values(p, action)
                            if q_val_p > max_q_p:
                                max_q_p = q_val_p
                        diff = abs(self.values[p] - max_q_p)
                        if diff > self.theta:
                            cua.update(p, -diff)
