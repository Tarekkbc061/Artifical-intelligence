# q_learning_agents.py
# ------------------
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


from game import *
from learning_agents import ReinforcementAgent
from feature_extractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - compute_value_from_q_values
        - compute_action_from_q_values
        - get_q_value
        - get_action
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.get_legal_actions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # guardem els Q-values en un Counter (per defecte retorna 0.0 si un parell estat-accio no s'ha vist)
        self.q_values = util.Counter()

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # retornem el Q-value guardat al nostre Counter per aquest estat i accio
        return self.q_values[(state, action)]


    def compute_value_from_q_values(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        accions_legals = self.get_legal_actions(state)

        # si no hi ha accions legals (estat terminal), el valor per definicio es 0.0
        if len(accions_legals) == 0:
            return 0.0

        # busquem el maxim Q-value de totes les accions legals disponibles
        max_q = float("-inf")
        for action in accions_legals:
            q_val = self.get_q_value(state, action)
            if q_val > max_q:
                max_q = q_val

        return max_q

    def compute_action_from_q_values(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        accions_legals = self.get_legal_actions(state)

        # si no hi ha accions possibles retornem None
        if len(accions_legals) == 0:
            return None

        # agafem el maxim Q-value que podem tenir en aquest estat
        max_q = self.compute_value_from_q_values(state)

        # fem una llista amb totes les accions que empaten amb aquest maxim Q-value
        millors_accions = []
        for action in accions_legals:
            if self.get_q_value(state, action) == max_q:
                millors_accions.append(action)

        # trenquem els empats de forma aleatoria amb random.choice per explorar millor
        return random.choice(millors_accions)

    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flip_coin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # si no hi ha accions legals retornem None
        if len(legal_actions) == 0:
            return None

        # tirem una moneda amb probabilitat epsilon per decidir si explorem o explotem
        if util.flip_coin(self.epsilon):
            # explorem: agafem una accio qualsevol a l'atzar
            action = random.choice(legal_actions)
        else:
            # explotem: escollim la millor accio segons els nostres Q-values
            action = self.compute_action_from_q_values(state)

        return action
        

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # apliquem l'equacio de Q-learning: (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max Q(s', a'))
        q_actual = self.get_q_value(state, action)
        valor_seguent = self.compute_value_from_q_values(next_state)

        mostra = reward + self.discount * valor_seguent
        self.q_values[(state, action)] = (1.0 - self.alpha) * q_actual + self.alpha * mostra

    def get_policy(self, state):
        return self.compute_action_from_q_values(state)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, state):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.get_action(self, state)
        self.do_action(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite get_q_value
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.feat_extractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def get_weights(self):
        return self.weights

    def get_q_value(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # calculem el Q-value multiplicant cada feature pel seu pes corresponent
        features = self.feat_extractor.get_features(state, action)
        q_val = 0.0

        for feat in features:
            q_val += self.weights[feat] * features[feat]

        return q_val
        

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # calculem la diferencia entre la mostra actual i el Q-value que teniem
        diferencia = (reward + self.discount * self.get_value(next_state)) - self.get_q_value(state, action)
        features = self.feat_extractor.get_features(state, action)

        # actualitzem el pes de cada feature amb la regla d'aprenentatge
        for feat in features:
            self.weights[feat] = self.weights[feat] + self.alpha * diferencia * features[feat]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
