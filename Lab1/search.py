# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem): #pila
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    #inicialitzar frontera, llista de visitats i s0
    frontier = util.Stack()
    visited = set()
    s0 = problem.get_start_state()
    frontier.push((s0, []))
    
    #bucle
    while not frontier.is_empty():
        #escollir el node d'asobre de la pila
        current_state, actions = frontier.pop()
        
        if problem.is_goal_state(current_state): return actions # si es goal acabem
        if current_state not in visited: 
            
            visited.add(current_state) #l'afegim a visitats
            #l'expandim
            for next_state, action, cost in problem.get_successors(current_state): # get_successors tambe retorna el cost. aixi recorrem tos els fills
                #si no ha estat visitat l'afegim a la llista
                if next_state not in visited:
                    
                    # guardem l'estat i el cami d'aquest node fill
                    next_actions = actions + [action] #sequencia d'accions fins a next_state
                    frontier.push((next_state, next_actions))
                    
    #si la cua esta buida i no hem trobat la solucio, retornem una llista budida --> fail
    return []
                
            
    
    util.raise_not_defined()



def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    # 1. Inicialización de la Frontera (Queue)
    # Para BFS necesitamos una cola FIFO (First In, First Out) provista por util.Queue.
    # Los primeros nodos en entrar serán los primeros en salir, garantizando la exploración por niveles.
    frontier = util.Queue()

    # 2. Conjunto de Estados Visitados (Explorados)
    # Guardamos únicamente los estados (por ejemplo coordenadas o configuraciones)
    # para evitar procesar el mismo estado múltiples veces y prevenir bucles infinitos.
    visited = set()

    # 3. Estado Inicial
    # Encolamos una tupla con (estado_actual, camino_de_acciones_acumulado).
    # Al principio, el camino es una lista vacía [] porque no se ha realizado ninguna acción.
    start_state = problem.get_start_state()
    frontier.push((start_state, []))

    # 4. Bucle principal de búsqueda
    while not frontier.is_empty():
        # Extraemos el nodo más superficial (menos profundo) de la cola
        current_state, actions = frontier.pop()

        # Comprobación de meta: si el estado actual es el objetivo, retornamos las acciones
        if problem.is_goal_state(current_state):
            return actions

        # Si el estado no ha sido expandido todavía
        if current_state not in visited:
            # Lo marcamos como visitado
            visited.add(current_state)

            # Expandimos el nodo obteniendo sus sucesores: (siguiente_estado, accion, coste)
            for next_state, action, cost in problem.get_successors(current_state):
                # Solo encolamos si el sucesor no ha sido visitado previamente
                if next_state not in visited:
                    # Concatenamos la nueva acción a una nueva lista (evitando modificar la original)
                    next_actions = actions + [action]
                    frontier.push((next_state, next_actions))

    # Si se vacía la cola y no se encontró la meta, retornamos lista vacía
    return []

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
