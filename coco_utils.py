import math
import itertools

import ray
from cvxopt import matrix, solvers
import numpy as np

import ray_utils

# Tell GLPK to produce no output, and timeout after 1 second
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF', 'tm_lim': 1000}


def maxmin_val_fast(A, solver="glpk"):
    """
    Function that computes the maxmin LP for a given matrix A
    A is the payoff matrix of one player in a zero-sum, two-player game
    GLPK is best solver to use for cvxopt. I tried Mosek, and it is much slower
    :param A:
    :param solver:
    :return:
    """
    # https://adamnovotnycom.medium.com/linear-programming-in-python-cvxopt-and-game-theory-8626a143d428
    num_vars = int(A.shape[0])
    # minimize matrix c
    c = [-1] + [0 for i in range(num_vars)]
    c = np.array(c, dtype="float")
    c = matrix(c)

    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T  # reformat each variable is in a row

    G *= -1  # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1])  # > 0 constraint for all vars
    new_col = [1 for i in range(A.shape[1])] + [0 for i in range(num_vars)]

    G = np.insert(G, 0, new_col, axis=1)  # insert utility column
    G = matrix(G)
    h = ([0 for i in range(A.shape[1])] + [0 for i in range(num_vars)])
    h = np.array(h, dtype="float")
    h = matrix(h)

    # contraints Ax = b
    A = [0] + [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)

    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)

    if sol['primal objective'] is not None:
        rval = -1 * sol['primal objective']
    else:
        rval = None

    return rval


class Game:
    """
    The Game class
    Implements a one-stage game with n players,
    as defined in final_report, Section 2.
    """

    def __init__(self, nplayers, payoffs, act_names=None):
        """
        # Constructor
        # nplayers : number of players in game
        # payoffs  : the payoff tensor for each player
        #          : payoffs[i] is payoff tensor for player i
        #          : all tensors must be same shape
        """

        if act_names is None:
            act_names = []
        self.nplayers = nplayers
        self.payoffs = np.array(payoffs)
        self.pmat = []
        for p in range(0, nplayers):
            self.pmat.append(payoffs[p])

        self.nStates = np.prod(payoffs[0].shape)
        self.states = self.gen_states()
        self.idxState = {}
        for i in range(len(self.states)):
            self.idxState[self.states[i]] = i
        self.act_names = act_names

    def print_state(self, state):
        """
        If act_names are supplied for each player,
        This prints the name of the action chosen by each player

        :param state:
        :return:
        """
        if len(self.act_names) == 0:
            return False
        tup = ()
        for p in range(self.nplayers):
            tup += (self.act_names[p][state[p]],)

    def coal_complement(self, coal):
        """
        :param coal:
        :return: the complement coalition of players to coal
        """
        new_coal = []
        for p in range(self.nplayers):
            if coal[p] == 1:
                new_coal.append(0)
            else:
                new_coal.append(1)

        return tuple(new_coal)

    def coco_values(self):
        """
        This function is how external code should
        access the Coco values. Returns a vector of
        length nplayers, with entry i being the Coco
        value for player i

        If we want to try a different
        def. for coco, can just replace which function
        is called within here.

        :return:
        """
        return self.compute_coco()

    def flatten_nested_tuple(self, data):
        """
        Flattens a tuple
        :param data:
        :return:
        """
        if isinstance(data, tuple):
            if len(data) == 0:
                return ()
            else:
                return self.flatten_nested_tuple(data[0]) + self.flatten_nested_tuple(data[1:])
        else:
            return data,

    def gen_states(self, shape_vector=None, p=-1, pact=-1):
        """
        This function generates all possible
        states. If no arguments are passed,
        by default it generates every joint
        action tuple. If a shape_vector is passed,
        it uses this shape vector to generate all
        tuples.

        One can also use the p, pact parameters to fix
        player p to have action act, and generate all
        joint actions with p being forced to have action act

        :param shape_vector:
        :param p:
        :param pact:
        :return:
        """
        # if (shape_vector == []):
        if shape_vector is None:
            shape_vector = []
        if len(shape_vector) == 0:
            shape_vector = self.payoffs[0].shape

        nplayers = len(shape_vector)
        if p == 0:
            states = [pact]
        else:
            states = list(range(0, shape_vector[0]))

        for i in range(1, nplayers):
            if p == i:
                states = list(itertools.product(states, [pact]))
            else:
                states = list(itertools.product(states, range(0, shape_vector[i])))

        for i in range(len(states)):
            states[i] = self.flatten_nested_tuple(states[i])

        return states

    def maxmax(self):
        """
        returns the action that maximizes the
        joint welfare. That is, all players' payoffs
        are added together, and the action that maximizes
        the sum of all payoffs is returned.

        :return:
        """
        mm = self.payoffs[0]
        for p in range(1, self.nplayers):
            mm += self.payoffs[p]
        act = np.unravel_index(mm.argmax(), mm.shape)

        return act

    def get_values_strategy(self, strat, payoffs=None):
        """
        Gets value of a set of mixed strategies
        for each player

        :param strat:
        :param payoffs:
        :return:
        """
        if payoffs is None:
            payoffs = self.payoffs
        nplayers = self.nplayers
        ss = self.states
        valPlayers = np.zeros(nplayers)
        try:
            for p in range(nplayers):
                valPlayers[p] = 0
                for x in ss:
                    valPlayers[p] += payoffs[p][x] * strat[self.idxState[x]]
        except:
            print("An error occurred. ")
            valPlayers = None

        return valPlayers

    def compute_coeff_coal(self, coal):
        size_c = np.array(coal)
        size_c = size_c.sum() - 1

        coeff = 1.0
        # coeff /= (math.factorial( self.nplayers - 1 ) / (math.factorial( size_c ) * math.factorial( self.nplayers - 1 - size_c )));
        coeff /= math.comb(self.nplayers - 1, size_c)
        coeff /= self.nplayers

        return coeff

    def compute_cc_coal_val(self, coal, payoffs=None):
        # coal is binary vector. Players in coalition have value 1
        # Player 1 will be coalition players
        if payoffs is None:
            payoffs = self.payoffs
        shapeActionsPlayer1 = []
        shapeActionsPlayer2 = []
        nplayers = self.nplayers
        coal_size = np.array(coal)
        coal_size = coal_size.sum()
        for p in range(nplayers):
            if coal[p] == 1:
                shapeActionsPlayer1.append(payoffs[0].shape[p])
            else:
                shapeActionsPlayer2.append(payoffs[0].shape[p])
        nActionsPlayer2 = int(np.prod(shapeActionsPlayer2))
        nActionsPlayer1 = int(np.prod(shapeActionsPlayer1))
        player1mat = np.zeros((nActionsPlayer1, nActionsPlayer2))
        player2mat = np.zeros((nActionsPlayer1, nActionsPlayer2))
        player1profit = np.zeros((nActionsPlayer1, nActionsPlayer2))
        p1states = self.gen_states(shapeActionsPlayer1)
        p2states = self.gen_states(shapeActionsPlayer2)
        n = nplayers

        for p1act in range(nActionsPlayer1):
            for p2act in range(nActionsPlayer2):
                origState = []
                p1actSep = list(p1states[p1act])
                p2actSep = list(p2states[p2act])
                for p in range(nplayers):
                    if coal[p] == 1:
                        origState.append(p1actSep[0])
                        p1actSep.pop(0)
                    else:
                        origState.append(p2actSep[0])
                        p2actSep.pop(0)
                origState = tuple(origState)
                for p in range(nplayers):
                    if coal[p] == 1:
                        player1mat[(p1act, p2act)] += payoffs[p][origState]
                    else:
                        player1mat[(p1act, p2act)] -= payoffs[p][origState]

                player2mat[(p1act, p2act)] = -1.0 * player1mat[(p1act, p2act)]

        mm = maxmin_val_fast(player1mat)

        rval = None
        if mm is not None:
            rval = mm
        return rval

    def compute_coco(self):
        """
        Our Coco definition for n players
        :return:
        """
        nplayers = self.nplayers
        pmat = self.pmat
        zsGames = []
        coco_vals = np.zeros(nplayers)

        coals = self.gen_states(np.full(nplayers, 2))
        coals.pop(0)  # discard the empty coalition
        v = np.full(np.full(nplayers, 2), None)
        for x in coals:
            # Check if we have already computed complement
            xc = self.coal_complement(x)
            coal_val = None
            if v[xc] is not None:
                coal_val = -1 * v[xc]
            else:
                coal_val = self.compute_cc_coal_val(x)

            if coal_val is not None:
                v[x] = coal_val
            else:
                return [None for p in range(nplayers)]
            coeff = self.compute_coeff_coal(x)
            for p in range(nplayers):
                if x[p] == 1:
                    coco_vals[p] += v[x] * coeff

        return coco_vals


@ray.remote
def compute_coco_group_distributed(data, num_players, cached_coco_values_groups):
    return compute_coco_group(data, num_players, cached_coco_values_groups)


def compute_coco_group(data, num_players, cached_coco_values_groups):

    # Tell GLPK to produce no output, and timeout after 1 second
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF', 'tm_lim': 1000}

    action_size = int(data.shape[-1] ** (1/num_players))
    sample_size = data.shape[0]
    new_shape = tuple(data.shape[:2]) + tuple([action_size] * num_players)
    new_data = np.reshape(data, new_shape)

    coco_values = np.zeros((sample_size, num_players))
    num_cached = 0
    num_not_cached = 0

    for idx in range(sample_size):
        cached_coco_values = cached_coco_values_groups[idx]
        if not np.isnan(cached_coco_values).any():
            coco_values[idx] = cached_coco_values
            num_cached += 1
        else:
            payoffs = new_data[idx]
            game = Game(nplayers=num_players, payoffs=payoffs)
            game_coco_values = game.coco_values()
            coco_values[idx] = game_coco_values
            num_not_cached += 1

    return coco_values


def compute_coco_distributed(data, num_actors, num_players, all_cached_coco_values):
    data_groups = np.array_split(data, num_actors)
    cached_coco_values_groups = np.array_split(all_cached_coco_values, num_actors)
    answer_refs = [compute_coco_group_distributed.remote(data_group, num_players, coco_group) for data_group, coco_group in zip(data_groups, cached_coco_values_groups)]
    answers = ray.get(answer_refs)
    stacked_answer = np.vstack(answers)

    return stacked_answer


if __name__ == '__main__':

    # test = np.array([np.full((8), n) for n in range(3)])
    # print(test)
    # new_test = np.reshape(test, (3, 2, 2, 2))
    # print(new_test)

    num_players = 4
    all_payoffs = np.random.random((512, 4, 1296))
    import time
    for i in range(1, 17):
        start = time.time()
        answer = compute_coco_distributed(data=all_payoffs, num_actors=i, num_players=num_players)
        end = time.time()
        print(i, end - start)

