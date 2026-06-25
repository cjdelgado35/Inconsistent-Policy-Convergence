import sys
import math
import numpy as np
from numpy.linalg import norm
import itertools
from mdp import get_probabilitiesVI
import mdp
from add import ADD
from random import *
from collections import deque
from statistics import mean
import time
from numpy import savetxt
from datetime import datetime

import copy


def initialize(C_max, u, k_g, mdp_obj, V_i, S, A, c=1):
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]
    n_states = len(G) + len(not_goal)

    C = np.arange(c + 1) + C_max
    V = np.full((n_states, C_max + c + 1), 0.0)
    pi = np.full((n_states, C_max + c + 1), None)
    VA = np.full((n_states, C_max + c + 1, len(A)), 0.0)
    V[G] = k_g
    not_goal_and_C = np.array([[i, j]
                               for i, j in itertools.product(not_goal, C)]).T
    not_goal_i, C_i = not_goal_and_C
    V[not_goal_i, C_i] = ((u(np.full(C.shape, np.inf)) -
                           u(C)) * np.ones((len(not_goal), len(C)))).flatten()

    return V, pi, VA


def gubs(C_max, u, k_g, mdp_obj, V_i, S, c=1):
    A = mdp.get_actions(mdp_obj)

    V, pi, VA = initialize(C_max, u, k_g, mdp_obj, V_i, S, A, c)
    itera = 0
    for C in reversed(range(C_max + 1)):
        for s in S:
            actions_results = np.array(
                [mdp.Q(s, C, a, u, V, V_i, mdp_obj, c) for a in A])
            VA[V_i[s], C] = actions_results
            if C == 0 and s == '10':
                print("Q:")
                print(actions_results)
            i_max = np.argmax(actions_results)
            pi[V_i[s], C] = A[i_max]
            V[V_i[s], C] = actions_results[i_max]
            itera +=1
            #print(V)
    print(A)
    return V, pi, itera, VA


def finite_gubs(S, A, C_max, V_i, H, mdp_obj, k_g, u, c=1):
    n_states = len(S)
    n_actions = len(A)
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]
    Q = np.zeros((n_states, C_max + c + 1, n_actions, H + 1))
    pi = np.full((n_states, C_max + c + 1, H + 1), None)
    # for i_a, _ in enumerate(A):
    #    Q[not_goal, :, i_a, H] = u(np.arange(C_max + c + 1)) - 1
    Q[G] = k_g

    for h in reversed(range(H)):
        for C in reversed(range(C_max + 1)):
            for s in S:
                i_s = V_i[s]
                for i_a, a in enumerate(A):
                    reachable = mdp.find_reachable(s, a, mdp_obj)
                    c_ = 0 if mdp_obj[s]['goal'] else c
                    s_a_cost = u(C + c_) - u(C)
                    Q[i_s, C, i_a, h] = s_a_cost + sum([
                        np.max(Q[V_i[s_['name']], C + c_, :, h + 1]) * s_['A'][a] for s_ in reachable])
                pi[i_s, C, h] = A[np.argmax(Q[i_s, C, :, h])]
    return Q, pi


def u(lamb, c): return np.exp(lamb * c)


def risk_sensitive(lamb, mdp_obj, V_i, S, c=1, epsilon=1e-3, n_iter=None):
    def u(c): return np.exp(lamb * c)

    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [i for i, s in mdp_obj.items() if not s['goal']]
    n_states = len(S)

    # initialize
    V = np.zeros(n_states, dtype=float)
    pi = np.full(n_states, None)
    P = np.zeros(n_states, dtype=float)
    V[G] = -np.sign(lamb)
    P[G] = 1
    A = np.array(mdp.get_actions(mdp_obj))

    i = 0

    P_not_max_prob = np.copy(P)
    while True:
        V_ = np.copy(V)
        P_ = np.copy(P)
        for s in not_goal:
            actions_results_p = np.array([
                np.sum([
                    P[V_i[s_['name']]] * s_['A'][a] for s_ in mdp.find_reachable(s, a, mdp_obj)
                ]) for a in A
            ])

            # set maxprob
            max_prob = np.max(actions_results_p)
            P_[V_i[s]] = max_prob
            A_max_prob = A[actions_results_p == max_prob]
            A_not_max_prob = A[actions_results_p != max_prob]
            not_max_prob_actions_results = np.array([
                np.sum([
                    P[V_i[s_['name']]] * s_['A'][a] for s_ in mdp.find_reachable(s, a, mdp_obj)
                ]) for a in A_not_max_prob
            ])

            # TODO -> Ajeitar esse cálculo abaixo, ta dando 0 sempre em alguns casos.
            #           - Talvez não esteja pegando as ações que não são maxprob corretamente
            # record maxprob obtained by actions that are in A_not_max_prob
            P_not_max_prob[V_i[s]] = P[V_i[s]] if len(not_max_prob_actions_results) == 0 else np.max(
                not_max_prob_actions_results)

            actions_results = np.array([
                np.sum([
                    u(c) * V[V_i[s_['name']]] * s_['A'][a] for s_ in mdp.find_reachable(s, a, mdp_obj)
                ]) for a in A_max_prob
            ])

            i_a = np.argmax(actions_results)
            if s == '6':
                print('EITA')
                print(' ', actions_results_p)
                print(' ', A_max_prob, i_a, A_max_prob[i_a])
                print(' ', actions_results, actions_results[i_a])
            if s == '43':
                print('EITA2')
                print(' ', actions_results_p)
            if s == '42':
                print('EITA3')
                print(' ', actions_results_p)
            V_[V_i[s]] = actions_results[i_a]
            pi[V_i[s]] = A_max_prob[i_a]

        v_norm = norm(V_ - V, np.inf)
        p_norm = norm(P_ - P, np.inf)

        P_diff = P_ - P_not_max_prob
        arg_min_p_diff = np.argmin(P_diff)
        min_p_diff = P_diff[arg_min_p_diff]

        if n_iter and i == n_iter:
            break
        #print('delta1:', v_norm, p_norm, v_norm + p_norm)
        #print('prob:', P, P_)
        #print('delta2:', P_diff, min_p_diff)
        if v_norm + p_norm < epsilon and min_p_diff >= 0:
            break
        V = V_
        P = P_
        i += 1

    print(f'{i} iterations')
    return V, P, pi


def get_X(V, V_i, lamb, S, A, mdp_obj, c=1):

    list_X = [
        (
            (s, a),
            (V[V_i[s]] - np.sum(
                np.fromiter(
                    (s_['A'][a] * u(lamb, c) * V[V_i[s_['name']]]
                     for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float))
             )
        )
        for (s, a) in itertools.product(S, A)
    ]

    X = np.array(list_X)

    return X[X.T[1] < 0]


def get_cmax(V, V_i, P, S, A, lamb, k_g, mdp_obj, c=1):
    X = get_X(V, V_i, lamb, S, A, mdp_obj)
    # print("X:", X)
    W = np.zeros(len(X))

    for i, ((s, a), x) in enumerate(X):
        # print('oi, ', s, a, x, P[V_i[s]], np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
        #                                               for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float), np.sum(np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
        #                                                                                                                               for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float)))
        denominator = k_g * (np.sum(np.fromiter((s_['A'][a] * P[V_i[s_['name']]]
                                                 for s_ in mdp.find_reachable(s, a, mdp_obj)), dtype=float)) - P[V_i[s]])
        if denominator == 0:
            W[i] = -np.inf
        else:
            # print('calc: ', -(1 / lamb), denominator, x /
            #      denominator, np.log(x / denominator))
            W[i] = -(1 / lamb) * np.log(
                x / denominator
            )

    print("W[s]:", [((s, a), W[i]) for i, ((s, a), x) in enumerate(X)])
    try:
        C_max = np.max(W[np.invert(np.isnan(W))])
    except:
        return 0
    if C_max < 0 or C_max == np.inf:
        return 0

    return int(np.ceil(C_max))


def exact_gubs(V_risk, P_risk, pi_risk, C_max, lamb, k_g, mdp_obj, V_i, S, A, c=1):
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    n_states = len(S)
    n_actions = len(A)

    V = np.zeros((n_states, C_max + 1))
    V_risk_C = np.zeros((n_states, C_max + 2))
    P = np.zeros((n_states, C_max + 2))
    pi = np.full((n_states, C_max + 2), None)

    #print(n_states, C_max)
    #print('V_risk:', V_risk)
    # print(V_risk_C.shape)
    # print(G)
    #V_risk_C[G, :] = k_g
    V_risk_C[G, :] = V_risk[G]
    P[G, :] = 1
    # print('V_risk_C antes antes:', V_risk_C)
    V_risk_C[:, C_max + 1] = V_risk.T
    # print('V_risk_C antes:', V_risk_C)
    P[:, C_max + 1] = P_risk.T
    pi[:, C_max + 1] = pi_risk.T

    n_updates = 0
    # print(V_risk)
    for C in reversed(range(C_max + 1)):
        #print(f'C = {C}')
        Q = np.zeros(n_actions)
        P_a = np.zeros(n_actions)
        for s in S:
            i_s = V_i[s]
            n_updates += 1
            for i_a, a in enumerate(A):
                c__ = 0 if mdp_obj[s]['goal'] else c
                c_ = C + c__
                reachable = mdp.find_reachable(s, a, mdp_obj)

                # Get value
                gen_q = [s_['A'][a] * V_risk_C[V_i[s_['name']], c_]
                         for s_ in reachable]
                #print(' gen_q:', gen_q, lamb, c__)
                Q[i_a] = u(lamb, c__) * \
                    np.sum(np.fromiter(gen_q, dtype=np.float))

                # Get probability
                gen_p = (s_['A'][a] * P[V_i[s_['name']], c_]
                         for s_ in reachable)
                P_a[i_a] = np.sum(
                    np.fromiter(gen_p, dtype=np.float)
                )
                # if s == '3':
                #print(C, s, P_a[i_a], Q[i_a], c_)

                #print(s, C)
            i_a_opt = np.argmax(u(lamb, C) * Q + k_g * P_a)
            a_opt = A[i_a_opt]
            #print('Q:', Q)
            #print('argmax:', u(lamb, C) * Q + k_g * P_a, i_a_opt, a_opt, A)
            pi[i_s, C] = a_opt

            P[i_s, C] = P_a[i_a_opt]
            V_risk_C[i_s, C] = Q[i_a_opt]
            #print('P:', P[i_s, C])
            #print('V_risk:', V_risk)
            #print('V_risk_C:', V_risk_C)
            ##print("Q: ", Q)
            # print(
            #    f'{i_s}, {C}, {V_risk_C[i_s, C]}, {V_risk_C[i_s]}, {P[i_s, C]}')
            V[i_s, C] = V_risk_C[i_s, C] + k_g * P[i_s, C]
            # print('  result:', V_risk_C[i_s, C], k_g * P[i_s, C],
            #      V_risk_C[i_s, C] + k_g * P[i_s, C], V[i_s, C])

    print("Updates:", n_updates)
    return V, P, pi

#Value Iteration###############################################################################

def initializeVI(mdp_obj, V_i, S, c=1):
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]
    n_states = len(G) + len(not_goal)

    V = np.full((n_states), 0.0)
    pi = np.full((n_states), None)

    return V, pi

def vi(mdp_obj, V_i, S, c=1):
    A = mdp.get_actions(mdp_obj)
    V, pi = initializeVI(mdp_obj, V_i, S, c)
    itera = 0
    res = sys.float_info.max
    epsilon = 0.00001;
    while res > epsilon:
        Vold = V.copy()
        for s in S:
            actions_results = np.array(
                [mdp.QVI(s, a, V, V_i, mdp_obj, c) for a in A])
            i_max = np.argmax(actions_results)
            pi[V_i[s]] = A[i_max]
            V[V_i[s]] = actions_results[i_max]
            itera +=1
        res = max(abs(Vold - V))
    return V, pi, itera

# SPUDD ########################################################################3

def CreateVADD(S, c):
    v = ADD.variable(S[0], c, c)
    vold = v
    for s in S[1:]:
        vs = ADD.variable(s, c, c)
        ADD.subvariable(vold, vs, vs)
        vold = vs
    return v

def SPUDD(mdp_obj, S, V_i, c=1):
    A = mdp.get_actions(mdp_obj)
    A_i = {A[i]: i for i in range(len(A))}
    CPTS = np.full((len(A), len(S)), None)
    CreateCPTS(mdp_obj, CPTS, S, V_i, A, A_i)
    Vdd = CreateVADD(S, 0)
    BE = sys.float_info.max
    epsilon = 0.00001;
    while BE > epsilon:
        VddNew = CreateVADD(S, sys.float_info.max)
        for a in A:
            Qdd = REGRESS(Vdd, a)
            VddNew = ADD.min(VddNew, Qdd)
        Diffdd = VddNew - Vdd
        BE = ADD.max(abs(Diffdd))
    return VddNew

def REGRESS(Vdd, a):
    Qdd = copy.deepcopy(Vdd)
    Qdd.CONVERTTOPRIMES()
    vars = Qdd.getVariables()
    #for X in vars:
    #    Qdd = Qdd x CPT(X)

def CreateCPTS(mdp_ojb, CPTS, S, V_i, A, A_i):
    parents = np.full((len(A), len(S)), None)
    for i in range(len(A)):
        for j in range(len(S)):
            parents[i, j] = []
    mdp.find_parents(mdp_ojb, S, V_i, A, A_i, parents)
    for s in S:
        for a in A:
            CPT = CreateCPT(mdp_ojb, a, s, V_i, A_i, parents)
            CPTS[A_i[a], V_i[s]] = CPT

def CreateCPT(mdp_obj, a, s, V_i, A_i, parents):
    parentstate = parents[A_i[a], V_i[s]]
    CPT = ADD.CPT(parentstate, s)
    SetVal(CPT)
    return CPT

def SetVal(CPT):
    stack = [(CPT, "HL", 0)]
    stackHL = (CPT._index, "HL",0)
    while stack:
        vertex = stack.pop()
        vertexHL = stackHL.pop()
        indexcount = vertexHL[2]
        index = vertexHL[0]
        if vertex.is_variable():
            stack.append(vertex._low)
            stack.append(vertex._high)
            stackHL.append((index, "L", indexcount))
            stackHL.append((vertex._index, "H", indexcount + 1))
        else:
            if indexcount == 1:
                vertex[0]._value = 5
            else:
                vertex[0]._value = 0

    print(CPT)


################# Q LEARNING


#def getInitialState(initialState):#Get an initial state
#    if initialState != 0:
#        return initialState

def getInitialState(initialState,nestados,c_max):#Get an initial state
    #if c_max:
    #    return str(np.random.randint(1, nestados)),np.random.randint(0, c_max)
    #else:
    #return str(np.random.randint(1, nestados)), np.random.randint(0, c_max+1)
    return str(np.random.randint(1, nestados)), np.random.randint(0, nestados)

def choose_action(actions, state, epsilon, Q):
    return actions[randrange(len(actions))] if (np.random.random() <= epsilon) else actions[np.argmax(Q[state])]

# Adaptive learning of Exploration Rate
def get_epsilon(t,MAXEPSILON,MINEPSILON,NEPISODES,epsilon):
    #return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))
    #return 0.9
    dec = (MAXEPSILON - MINEPSILON) / NEPISODES
    return max(epsilon - dec, MINEPSILON)

#Adaptive learning of Learning Rate
def get_alpha(t, max_alpha,min_alpha,NEPISODES, alpha):
    #ada_divisor = 25
    #return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))
    dec = (max_alpha - min_alpha) / NEPISODES
    return max(alpha - dec, min_alpha)
    #return 0.78
    #return 0.1

# Updating Q-value of state-action pair based on the update equation
def update_q(Q, state_old, action, actiond, reward, state_new, alpha, gamma, f):
    if(state_old == 12):
        print(Q[state_old])
        f.write(str(Q[state_old][0]) + "," + str(Q[state_old][1]) + "," + str(Q[state_old][2]) + "," + str(Q[state_old][3]) + "\n")
        #res = Q[state_old][action] + alpha * (reward + gamma * np.max(Q[state_new]) - Q[state_old][action])
        #print("Q(1)" + "(b)" + " " + str(state_new+1) + " " + " = " + str(Q[state_old][action]) + " + " + str(alpha) + " * " + str(reward) + " + " + str(gamma) + " * " + str(np.max(Q[state_new])) + " - " + str(Q[state_old][action]) + " = " + str(res))
    Q[state_old][action] += alpha * (reward + gamma * np.max(Q[state_new]) - Q[state_old][action])

def Qlearning(mdp_obj, S, V_i, n_episodes, min_alpha, min_epsilon, gamma, initialState):
    f = open('valoresQ', 'w')
    A = mdp.get_actions(mdp_obj)
    f.write(str(A[0]) + "," + str(A[1]) + "," + str(A[2]) + "," + str(A[3]) + "\n")
    V_a = {A[i]: i for i in range(len(A))}
    Q = np.zeros((len(S), len(A)))
    alpha = 1
    epsilon = 1
    for e in range(n_episodes):
        done = False
        current_state = getInitialState(initialState)
        alpha = get_alpha(e,min_alpha, n_episodes, alpha)
        epsilon = get_epsilon(e,min_epsilon, n_episodes, epsilon)
        totalreward = 0
        while not done:
            action = choose_action(A, V_i[current_state], epsilon, Q)
            new_state, reward, done = mdp.step(mdp_obj, current_state, action)
            update_q(Q, V_i[current_state], V_a[action], action, reward, V_i[new_state], alpha, gamma, f)
            current_state = new_state
            totalreward += reward
        #print("reset")
        print("episodio:" + str(e))
        #print(totalreward)
        print(alpha)
        print(epsilon)
        #print(epsilon)
        #print(Q)
    #Get Polity
    pi = np.full((len(S)), None)
    for s in S:
        pi[V_i[s]] = A[np.argmax(Q[V_i[s]])]
    print(pi)
    f.close()

################# Q GUBS

def choose_actionQGubs(actions, state, c_current, epsilon, Q):
    return actions[randrange(len(actions))] if (np.random.random() <= epsilon) else actions[np.argmax(Q[state][c_current])]

def update_qgubs(V, pi, u, Q, state_old, c_current, action, reward, state_new, alpha, gamma, A, S, C_max, QP, QC, f):
    #if S[state_old]  == '13' and c_current == 0:
    #    print(Q[state_old][c_current])
    #    f.write(str(Q[state_old][c_current][0]) + "," + str(Q[state_old][c_current][1]) + "," + str(Q[state_old][c_current][2]) + "," + str(Q[state_old][c_current][3]) + "\n")
    s_a_cost = u(c_current - reward) - u(c_current)
    Q[state_old][c_current][action] += alpha * (s_a_cost + gamma * np.max(Q[state_new][c_current-reward]) - Q[state_old][c_current][action])
    QP[state_old][c_current][action] += Q[state_old][c_current][action]
    QC[state_old][c_current][action] += 1
    pi[state_old][c_current] = A[np.argmax(Q[state_old][c_current])]
    V[state_old][c_current] = np.max(Q[state_old][c_current])
    if S[state_old]  == '13' and c_current == 0:
        print(Q[state_old][c_current])

def initializegubs(C_max, u, k_g, mdp_obj, V_i, S, A, c):
    V, pi = initialize(C_max, u, k_g, mdp_obj, V_i, S, c)
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    Q = np.zeros((len(S), C_max + c + 1, len(A)))
    QP = np.zeros((len(S), C_max + c + 1, len(A)))
    QC = np.zeros((len(S), C_max + c + 1, len(A)))
    C = np.arange(C_max + c + 1)
    #C = np.arange(c + 1) + C_max
    not_goal = [V_i[i] for i, s in mdp_obj.items() if not s['goal']]
    nA = np.arange(len(A))
    not_goal_and_C = np.array([[i, j, k]
                               for i, j, k in itertools.product(not_goal, C, nA)]).T
    not_goal_i, C_i, A_i = not_goal_and_C
    C = np.repeat(C, len(A))
    vals = ((u(np.full(C.shape, np.inf)) - u(C)) * np.ones((len(not_goal), len(C)))).flatten()
    Q[not_goal_i, C_i, A_i] = vals
    Q[G] = k_g
    return Q, V, pi, QP, QC

def QMean(QP,QC,S,C_max,c,A):
    for i in range(len(S)):
        for j in range(C_max + c + 1):
            for k in range(len(A)):
                if QC[i][j][k] >= 0:
                    QP[i][j][k] = QP[i][j][k] / QC[i][j][k]

def Qgubs(mdp_obj, S, V_i, n_episodes, min_alpha, min_epsilon, gamma, initialState, C_max, u, k_g, c=1):
    f = open('valoresQ', 'w')
    A = mdp.get_actions(mdp_obj)
    f.write(str(A[0]) + "," + str(A[1]) + "," + str(A[2]) + "," + str(A[3]) + "\n")
    V_a = {A[i]: i for i in range(len(A))}
    Q, V, pi, QP, QC = initializegubs(C_max, u, k_g, mdp_obj, V_i, S, A, c)
    alpha = 1
    epsilon = 1
    for e in range(n_episodes):
        done = False
        current_state = getInitialState(initialState)
        alpha = get_alpha(e, min_alpha, n_episodes, alpha)
        epsilon = get_epsilon(e, min_epsilon, n_episodes, epsilon)
        C = 0
        while not done:
            if C > C_max:
                break
            action = choose_actionQGubs(A, V_i[current_state], C, epsilon, Q)
            new_state, reward, done = mdp.step(mdp_obj, current_state, action)
            update_qgubs(V, pi, u, Q, V_i[current_state], C, V_a[action], reward, V_i[new_state], alpha, gamma, A, S, C_max, QP, QC, f)
            C += (reward * -1)
            current_state = new_state

        print("episodio:" + str(e))
        print(alpha)
        print(epsilon)
        #print(V)
    QMean(QP,QC,S,C_max,c,A)
    print(QP[9][1])
    print(A)
    f.close()
    return V, pi

# QeGbus
def choose_actionQrisk_sensitive(actions, state, epsilon, pi):
    return actions[randrange(len(actions))] if ((np.random.random() <= epsilon) or (pi[state]== None)) else pi[state]

def not_cycle_QP(mdp_obj, T_, G, V_i, V_a, A, QP, cycles, state_old): #get not cycle QP
    global iteraciones
    if state_old in G: #goal states have probabilitie 1
        return
    #all_reachable_from_state_old = mdp_obj[str(state_old+1)]['Adj'] #suppose model of state trasitions
    for action in A:
        L = find_reachable(state_old, V_a[action], T_)
        '''L = list(filter(
            lambda obj_s_: action in obj_s_['A'],
            all_reachable_from_state_old
        ))'''
        exist = False
        for s in L:#verify if exist one path in state that not form cycle
            #if len(cycles[V_i[s['name']]]) == 0:
            #    exist = True
            #    break
            if s in G: # if s is goal, exist path to goal
                exist = True
                break
            for path in cycles[s]:#cycles[V_i[s['name']]]:
                iteraciones += 1
                List = list(path)
                List.insert(0,s)#V_i[s['name']]) #complete de path with the target state in the path of target state
                if state_old not in List:
                    exist = True
                    break
            if exist: break
        if not exist:
            QP[V_a[action]] = 0

def updateCycles(mdp_obj, T_, V_i, V_a, cycles, s_source, action, pi):
    global iteraciones
    #all_reachable_from_s_source = mdp_obj[str(s_source + 1)]['Adj']  # suppose model of state trasitions
    L = find_reachable(s_source, V_a[action], T_)
    '''L = list(filter(
            lambda obj_s_: action in obj_s_['A'],
            all_reachable_from_s_source
    ))'''
    cycles[s_source] = set()
    orden = 1
    for s in L:  # add cycles of target state
        paths = cycles[s].copy()#V_i[s['name']]].copy()
        for path in paths:
            List = list()# path
            for e in path:
                iteraciones += 1
                List.append(e)
            #if s not in List:#if V_i[s['name']] not in List: # do not include infinity cycles - add
            #    List.insert(0,s)#V_i[s['name']]) # must include the target state in the path
            List.insert(0, s)
            if s_source in List: #if get a cycle do not include because this path is inconsistent
                continue
            cycles[s_source].add(frozenset(List)) # add path to the state
            addPath(s_source, s, orden, pi[s_source], frozenset(List))
            orden += 1
        if len(cycles[s]) == 0:#V_i[s['name']]]) == 0: # at least one path to the target had been discovered
            List = list()  # path
            List.append(s)#V_i[s['name']])
            cycles[s_source].add(frozenset(List)) # add path to the state
            addPath(s_source, s, orden, pi[s_source], frozenset(List))


def update_TR(s, a, news, reward, QN, N, RT, T):
    QN[s][a][news] += 1
    N[s][a] += 1
    RT[s][a] += reward * -1
    T[s][a][news] = QN[s][a][news] / N[s][a]#not exact

def calculateTR(T, R, S, A, QN, N, RT, V_i, V_a):
    T_= T.copy()
    '''for s in S:
        for a in A:
            for news in S:
                #R[s][a] = R[s][a] + alpha*((RT[s][a]/N[s][a]) - R[s][a])
                T_[V_i[s]][V_a[a]][V_i[news]] = QN[V_i[s]][V_a[a]][V_i[news]] / N[V_i[s]][V_a[a]]'''
    return T_

def find_reachable(s, a, T):
    all_states = list(T[s][a])
    return [i for i in range(len(all_states)) if all_states[i] > 0]

update = 0
updates1 = np.zeros((2000000, 4))
def addUpdate(e):
    global update, updates1
    if update < 2000000:
        updates1[update] = e
    update += 1

updatecycle = 0
updates1cycle = []
def addPath(s, s2, orden, p, path):
    pt = ""
    for e in path:
        pt = pt + str(e) + ","
    global updatecycle, updates1cycle
    #if updatecycle < 2000000:
        #updates1cycle[update] = ["s", "p", "1,2"]
    updates1cycle.append("Estado: "+str(s)+"Siguiente: "+str(s2)+","+p+","+str(orden)+","+pt+"\n")
    updatecycle += 1

def update_Qrisk_sensitive(mdp_obj, T_, G, V_i, pi, width, nactions, nstates, u, QP, QV, state_old, action, cost, state_new, alpha, A, V_a, P, P_, P_act_,V , V_, W, CMax, sMax, aMax, k_g, lamb, initialState , epsilon, e, cycles, laste, begin, step):#, P_not_max_prob, step):
    #if state_old == initialState: #and c_current == 0:
        #print(QP[state_old])
        #mdp.render(QP, nstates, A, nactions, width)
   #     print(str(e) + "," + str(alpha) + "," + str(epsilon))

    # To avoid error cycles, re-initialize other QP values for new calculation
    '''oldP = QP[state_old][action]
    newP = QP[state_old][action] + alpha * (P[state_new] - QP[state_old][action])
    if action == P_act_[state_old] and P_[state_old] > newP:# If max probability is decreasing
        for a in A:
            if V_a[a] != P_act_[state_old] and QP[state_old][V_a[a]] > newP and QP[state_old][V_a[a]] <= oldP:#if P is between old and new Probabilities, could be incorrect for cycle
                QP[state_old][V_a[a]] = newP#re-initialize Probability
                print("re-inicializa", str(state_old), a)'''

    QP[state_old][action] += alpha * (P[state_new] - QP[state_old][action])
    action_max_prob = np.argmax(QP[state_old])
    P_act_[state_old] = action_max_prob # To avoid error cycles
    # get QP without cycles
    not_cycle_QP(mdp_obj, T_, G, V_i, V_a, A, QP[state_old], cycles, state_old)# To avoid error cycles
    max_prob = np.max(QP[state_old])
    P_[state_old] = max_prob
    A_max_prob = A[QP[state_old] == max_prob]

    #A_not_max_prob = A[QP[state_old] != max_prob]
    #not_max_prob_actions_results = np.array([QV[state_old][a] for a in A_not_max_prob])
    #P_not_max_prob[state_old] = P[state_old] if len(not_max_prob_actions_results) == 0 else np.max(
    #    not_max_prob_actions_results)

    QV[state_old][action] += alpha * (u(cost) * V[state_new] - QV[state_old][action])
    actions_results = np.array([
        QV[state_old][V_a[a]] for a in A_max_prob
    ])
    i_a = np.argmax(actions_results)
    V_[state_old] = actions_results[i_a]
    pi[state_old] = A_max_prob[i_a]
    updateCycles(mdp_obj, T_, V_i, V_a, cycles, state_old, pi[state_old], pi)# To avoid error cycles

    # Calcular W(s,a) y actualizar CMax
    numerator = V_[state_old] - QV[state_old][action]
    if numerator < 0:#Si pertenece a X
        denominator = k_g * (QP[state_old][action] - P_[state_old])
        if denominator == 0:
            W[state_old][action] = -np.inf
        else:
            W[state_old][action] = -(1 / lamb) * np.log(numerator / denominator)
        if state_old == sMax and action == aMax and np.ceil(W[state_old][action]) < CMax:
            if not np.isnan(W[state_old][action]):
                CMax = W[state_old][action]
            else:
                CMax = -np.inf
                sMax = -1
                aMax = -1
        if not np.isnan(W[state_old][action]) and W[state_old][action] > CMax:
            CMax = W[state_old][action]
            sMax = state_old
            aMax = action
            #print(sMax,aMax,CMax,V_[state_old],QV[state_old][action],QP[state_old][action],P_[state_old])
    else:
        W[state_old][action] = -np.inf # si ya no pertenece a X entonces el W no debe considerarse
        if state_old == sMax and action == aMax:#si el cmax ya no pertenece a X
            CMax = -np.inf
            sMax = -1
            aMax = -1
    if CMax < 0 or CMax == np.inf:
        CMax = 0
    else:
        CMax = int(np.ceil(CMax))

    #if e != laste[0] and e % 4000 == 0:
    #if step % 8 == 0:
    #    end = time.time()
    #    addUpdate([step,  P_[initialState], (end - begin)])
    #    #laste[0] = e
    return CMax,sMax,aMax

iteraciones = 0
def eGubsQlearning(width, lamb, mdp_obj, V_i, S, n_episodes, n_episodese, min_alpha, min_epsilon, initialState, k_g, epsilonRS=1e-3, c=1, ncmax=10):
    def u(c): return np.exp(lamb * c)
    global updates1, update, updatecycle, updates1cycle, iteraciones
    iteraciones = 0
    updates1 = np.zeros((2000000, 4))
    updates1cycle = []
    update = 0
    updatecycle = 0
    begin = time.time()
    #Max = np.zeros((n_episodes,2))#Para prueba
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    n_states = len(S)
    cycles = [0 for i in range(n_states)] #For control cycles
    for i in range(len(cycles)):#For control cycles
        cycles[i] = set()#For control cycles
        #cycles[i].append(i)#himself
    primera = True
    # initialize
    V = np.zeros(n_states, dtype=float)
    pi = np.full(n_states, None)
    P = np.zeros(n_states, dtype=float)
    P_act = np.zeros(n_states, dtype=float) #Error cycles
    V[G] = -np.sign(lamb)
    P[G] = 1
    act = mdp.get_actions(mdp_obj)
    A = np.array(act)
    n_actions = len(A)
    V_a = {A[i]: i for i in range(n_actions)}
    alpha = 0.5#0.04#0.5
    alphaini = 0.5#0.04#0.5
    epsilon = 1
    maxepsilon = 1
    alphae = 0.5#0.04#0.5#0.04#0.5
    alphaeini = 0.5#0.04#0.5#0.04  # 0.5
    epsilone = 1
    maxepsilone = 1
    QP = np.zeros((n_states, n_actions))
    QV = np.zeros((n_states, n_actions))
    W = np.zeros((n_states, n_actions))#Unificado
    CMax = 0#Unificado Current
    C_Max = 0#Unificado New
    C_Max2 = 0  # Unificado New
    sMax = -1#Unificado
    aMax = -1#Unificado
    QV[G] = -np.sign(lamb)
    QP[G] = 1
    #initialize eQGUBS
    ReplayX = np.zeros((n_states, CMax + c + 1))  # cálculo de probabilidad
    Replay = np.zeros(n_states)  # Solo para depuracion
    ReplayX[G, :] = 1
    Replay[G] = 1
    Ve = np.zeros((n_states, CMax + 1))
    V_risk_Ce = np.zeros((n_states, CMax + 2))
    Pe = np.zeros((n_states, CMax + 2))
    pie = np.full((n_states, CMax + 2), None)
    Pe[G, :] = 1
    V_risk_Ce[G] = -np.sign(lamb)# Probabilidad
    V_a = {A[i]: i for i in range(n_actions)}
    Qe = np.zeros((n_states, CMax + c + 1, n_actions))
    QPe = np.zeros((n_states, CMax + c + 1, n_actions))
    Qe[G] = -np.sign(lamb)# Probabilidad
    QPe[G] = 1# Probabilidad
    QN = np.zeros((len(S), len(A), len(S)))  # times s,a,s'
    N = np.zeros((len(S), len(A)))  # times s,a
    RT = np.zeros((len(S), len(A)))  # total reward
    T = np.zeros((len(S), len(A), len(S)))  # transitions
    R = np.zeros((len(S), len(A)))  # rewards
    donee = True
    memory = deque(maxlen=ncmax)
    allcmax = []
    allcmax2 = []
    print(A)
    ee = 0#episodes eQGubs
    e = 0#episodes lexicografic
    laste = [-1]
    step = 0
    while e < n_episodes or ee < n_episodese:
        #dual
        done = False
        V_ = np.copy(V)
        P_ = np.copy(P)
        V_risk_Ce_ = np.copy(V_risk_Ce)
        Pe_ = np.copy(Pe)
        P_act_ = np.copy(P_act) #Error cycles
        current_state, C = getInitialState(initialState, len(S), 0)
        #alpha = get_alpha(e, alphaini, min_alpha, n_episodes-(0.2*n_episodes), alpha)
        alpha = get_alpha(e, alphaini, min_alpha, n_episodes, alpha)
        epsilon = get_epsilon(e, maxepsilon, min_epsilon, n_episodes, epsilon)
        while not done:
            step += 1
            #if step % 8 == 0:
            #    end = time.time()
            #    addUpdate([step, P[V_i[initialState]], (end - begin), iteraciones])
                #addUpdate([step, Pe[V_i[initialState], 0], (end - begin), iteraciones])
            action = choose_actionQrisk_sensitive(A, V_i[current_state], epsilon, pi)
            new_state, reward, done = mdp.step(mdp_obj, current_state, action)
            #T_ = calculateTR(T, R, S, A, QN, N, RT, V_i, V_a)
            C_Max,sMax,aMax = update_Qrisk_sensitive(mdp_obj, T, G, V_i, pi, width, n_actions, n_states, u, QP, QV, V_i[current_state], V_a[action], reward*-1, V_i[new_state], alpha, A, V_a, P, P_, P_act_, V, V_, W, C_Max, sMax, aMax, k_g, lamb, V_i[initialState], epsilon, e, cycles, laste, begin, step)#, P_not_max_prob)
            update_TR(V_i[current_state], V_a[action], V_i[new_state], reward, QN, N, RT, T)
            current_state = new_state
            #memory.append(C_Max)
            #C_Max = math.floor(mean(memory))#obtiene la media de 10 cmax
            #C_Max = 45#45solo prueba
            #allcmax.append((int(sMax), int(aMax), C_Max))
            #C_Max2 = np.max(W) if np.isnan(np.max(W)) else math.floor(np.max(W))  # Solo para prueba
            stateCmax = np.ceil((np.argmax(W) + 1 ) / n_actions)  # Solo para prueba
            actionCmax = np.argmax(W) % n_actions # Solo para prueba
            #allcmax2.append((int(stateCmax), int(actionCmax), C_Max2))
            # EQGubs
            if C_Max > CMax:  # Redimensionar matrices, deberia usarse algo más eficiente que el hstack
                d = int(C_Max - CMax)
                ReplayX = np.hstack((ReplayX, np.zeros((n_states, d))))
                ReplayX[G, CMax + c + 1:] = 1
                Ve = np.hstack((Ve, np.zeros((n_states, d))))
                V_risk_Ce = np.hstack((V_risk_Ce, np.zeros((n_states, d))))
                Pe = np.hstack((Pe, np.zeros((n_states, d))))
                V_risk_Ce_ = np.hstack((V_risk_Ce_, np.zeros((n_states, d))))
                Pe_ = np.hstack((Pe_, np.zeros((n_states, d))))
                pie = np.hstack((pie, np.full((n_states, d), None)))
                Pe[G, CMax + 2:] = 1
                V_risk_Ce[G, CMax + 2:] = -np.sign(lamb)# probabilidad
                Pe_[G, CMax + 2:] = 1
                V_risk_Ce_[G, CMax + 2:] = -np.sign(lamb)  # probabilidad
                V_a = {A[i]: i for i in range(n_actions)}
                Qe = np.hstack((Qe, np.zeros((n_states, d, n_actions))))
                QPe = np.hstack((QPe, np.zeros((n_states, d, n_actions))))
                Qe[G, CMax + 2:] = -np.sign(lamb)# probabilidad
                QPe[G, CMax + 2:] = 1# probabilidad
                CMax = C_Max
            if donee:#Cambio de episodio eQGubs
                current_statee, C = getInitialState(initialState, n_states, C_Max)
                # if e > n_episodes / 2:
                ee += 1
                donee = False
                #alphae = get_alpha(ee, alphaeini, min_alpha, n_episodese - (0.3 * n_episodese), alphae)
                alphae = get_alpha(ee, alphaeini, min_alpha, n_episodese, alphae)
                epsilone = get_epsilon(ee, maxepsilone, min_epsilon, n_episodese, epsilone)
            if C <= C_Max:
                actione = choose_actionQGubs(A, V_i[current_statee], C, epsilon, Qe)
                new_statee, rewarde, donee = mdp.step(mdp_obj, current_statee, actione)
                if C + ((-1)*rewarde) > C_Max:#es un dead end #Probabilidad
                        V_risk_Ce[V_i[new_statee], C + ((-1) * rewarde)] = 0 #Probabilidad
                        Pe[V_i[new_statee], C + ((-1) * rewarde)] = 0 #Probabilidad
                if donee and rewarde == 0:#Si el siguiente estado es objetivo el valor es 1, si es deadend el valor es 0 ya en la tabla
                    V_risk_Ce[V_i[new_statee], C + ((-1)*rewarde)] = -np.sign(lamb)
                update_qegubs(C_Max, Ve, V_risk_Ce, V_risk_Ce_, Pe, Pe_, pie, Qe, QPe, P, V, V_i[current_statee], C, V_a[actione], rewarde, V_i[new_statee], alphae,
                              A, lamb, k_g, S, initialState, epsilone, ee, e, ReplayX, Replay)

                C += (rewarde * -1)
                current_statee = new_statee
            else:
                donee = True
        V = V_
        P = P_
        V_risk_Ce = V_risk_Ce_
        Pe = Pe_
        P_act = P_act_ #Error cycles
        e += 1
        end = time.time()
        addUpdate([e, P[V_i[initialState]], (end - begin), iteraciones])
        #C_maxReal = get_cmaxQLeaning(V, V_i, P, S, act, QV, QP, lamb, k_g,W, mdp_obj)#solo para prueba
        #Max[e][0] = C_Max  # Solo para prueba
        #Max[e][1] = C_maxReal  # Solo para prueba
    print(A)
    savetxt(str(k_g) + '-' + str(i) + 'updates1Cycles4x10lexico' + str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.csv', updates1, delimiter=',')
    return Ve, Pe, pie, C_Max, V, pi, P, allcmax, allcmax2, stateCmax, actionCmax

def LexMonteCarlo(lamb, mdp_obj, V_i, S, n_episodes, min_epsilon, initialState):
    def u(c):
        return np.exp(lamb * c)
    global updates1, update
    updates1 = np.zeros((2000000, 3))
    update = 0
    begin = time.time()
    n_states = len(S)
    # initialize
    pi = np.full(n_states, None)
    P = np.zeros(n_states, None)
    act = mdp.get_actions(mdp_obj)
    A = np.array(act)
    n_actions = len(A)
    V_a = {A[i]: i for i in range(n_actions)}
    epsilon = 0.3#1
    maxepsilon = 0.3
    print(A)
    rewards = np.zeros((n_states, n_actions))
    rewardsAvg = np.zeros((n_states, n_actions))
    utilities = np.zeros((n_states, n_actions))
    utilitiesAvg = np.zeros((n_states, n_actions))
    times = np.zeros((n_states, n_actions))
    generalStep = 0
    e = 0#episodes lexicografic
    while e < n_episodes:
        #dual
        episode = []
        statesepisode = []
        first = np.zeros((n_states, n_actions))#first ocurrency of s,a in episode
        done = False
        current_state, C = getInitialState(initialState, len(S), 0)
        epsilon = get_epsilon(e, maxepsilon, min_epsilon, n_episodes, epsilon)
        step = 0
        while not done: #rollout
            generalStep += 1
            step += 1
            action = choose_actionQrisk_sensitive(A, V_i[current_state], epsilon, pi)
            new_state, reward, done, cost = mdp.stepMC(mdp_obj, current_state, action)
            episode.append([step, V_i[current_state], V_a[action], reward, cost])
            statesepisode.append([generalStep, V_i[current_state]])
            if first[V_i[current_state]][V_a[action]] == 0:
                first[V_i[current_state]][V_a[action]] = step
                if current_state == initialState:
                    print(str(rewardsAvg[V_i[current_state]]) + " - " + str(e) + "," + str(epsilon))
            current_state = new_state
        totalreward = 0
        totalcost = 0
        while len(episode)>0:
            step = episode.pop()
            state = step[1]
            action = step[2]
            totalreward += step[3]
            if totalcost == -1:
                totalcost = -1
            else:
                if step[4] == 1:
                    totalcost += step[4]
                else:
                    totalcost = -1
            if first[state][action] == step[0]:#if first ocurrency
                times[state][action] += 1
                rewards[state][action] = rewards[state][action] + totalreward
                rewardsAvg[state][action] = rewards[state][action] / times[state][action]
                if totalcost != -1:
                    utilities[state][action] = utilities[state][action] + u(totalcost)
                utilitiesAvg[state][action] = utilities[state][action] / times[state][action]
        while len(statesepisode) > 0:
            state = statesepisode.pop()
            stepNumber = state[0]
            state = state[1]
            max_prob = np.max(rewardsAvg[state])
            P[state] = max_prob
            A_max_prob = A[rewardsAvg[state] == max_prob]

            actions_results = np.array([
                utilitiesAvg[state][V_a[a]] for a in A_max_prob
            ])
            i_a = np.argmax(actions_results)
            pi[state] = A_max_prob[i_a]
            if stepNumber % 100 == 0:
                end = time.time()
                addUpdate([stepNumber, P[V_i[initialState]], (end - begin)])
        e += 1
    print(A)
    savetxt('updates1Cycles3x15' + str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.csv', updates1, delimiter=',')
    return utilitiesAvg, pi, rewardsAvg, P

def get_XQLeaning(V, V_i, QV, S, A):
    V_a = {A[i]: i for i in range(len(A))}
    list_X = [
        (
            (s, a),
            (V[V_i[s]] - QV[V_i[s]][V_a[a]])
        )
        for (s, a) in itertools.product(S, A)
    ]

    X = np.array(list_X)

    return X[X.T[1] < 0]


def get_cmaxQLeaning(V, V_i, P, S, A, QV, QP, lamb, k_g, W, mdp_obj):
    V_a = {A[i]: i for i in range(len(A))}
    X = get_XQLeaning(V, V_i, QV, S, A)
    # print("X:", X)
    W2 = np.zeros(len(X))

    for i, ((s, a), x) in enumerate(X):
        denominator = k_g * (QP[V_i[s]][V_a[a]] - P[V_i[s]])
        if denominator == 0:
            W2[i] = -np.inf
        else:
            W2[i] = -(1 / lamb) * np.log(
                x / denominator
            )

    #print("W[s]:", [((s, a), W[i]) for i, ((s, a), x) in enumerate(X)])
    try:
        C_max = np.max(W2[np.invert(np.isnan(W2))])
    except:
        return 0
    if C_max < 0 or C_max == np.inf:
        return 0

    return int(np.ceil(C_max))

def u(lamb, c): return np.exp(lamb * c)

def update_qegubs(C_Max, V, V_risk_C, V_risk_C_, P, P_, pi, Q, QP, Pl, Vl, state_old, c_current, action, reward, state_new, alpha, A, lamb, k_g, S, initialState , epsilon, ee, e, Replayx, Replay):
    if S[state_old] == initialState and c_current == 0:
        print(str(QP[state_old][c_current]) + " - " + str(e) + "," + str(ee) + "," + str(alpha) + "," + str(epsilon)+ "," + str(C_Max))
    Replayx[state_old][c_current] += 1
    Replay[state_old] += 1
    #s_a_cost = u(c_current - reward) - u(c_current)
    c_ = c_current + ((-1)*reward) #reward es costo negativo 0 si es Goal y -1 en otro caso
    if c_<= C_Max:
        Q[state_old][c_current][action] += alpha * ((u(lamb, ((-1)*reward)) * V_risk_C[state_new][c_]) - Q[state_old][c_current][action])# Get value
        QP[state_old][c_current][action] += alpha * (P[state_new][c_] - QP[state_old][c_current][action])# Get probability
    else: #use lexicografic polity for the next state
        Q[state_old][c_current][action] += alpha * (
                    (u(lamb, ((-1) * reward)) * Vl[state_new]) - Q[state_old][c_current][action])  # Get value
        QP[state_old][c_current][action] += alpha * (
                    Pl[state_new] - QP[state_old][c_current][action])  # Get probability

    i_a_opt = np.argmax(u(lamb, c_current) * Q[state_old][c_current] + k_g * QP[state_old][c_current])
    a_opt = A[i_a_opt]
    pi[state_old, c_current] = a_opt
    P_[state_old, c_current] = QP[state_old][c_current][i_a_opt]
    V_risk_C_[state_old, c_current] = Q[state_old][c_current][i_a_opt]
    V[state_old, c_current] = V_risk_C[state_old][c_current] + k_g * P[state_old, c_current]


def Qegubs(V_risk, P_risk, pi_risk, C_max, lamb, k_g, mdp_obj, V_i, S, A, n_episodes, min_alpha, min_epsilon, initialState, c=1):
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    n_states = len(S)
    n_actions = len(A)

    ReplayX = np.zeros((len(S), C_max + c + 1))  # cálculo de probabilidad
    Replay = np.zeros(len(S))  # Solo para depuracion
    ReplayX[G, :] = 1
    Replay[G] = 1

    V = np.zeros((n_states, C_max + 1))
    V_risk_C = np.zeros((n_states, C_max + 2))
    P = np.zeros((n_states, C_max + 2))
    pi = np.full((n_states, C_max + 2), None)

    #V_risk_C[G, :] = V_risk[G]
    V_risk_C = (np.repeat(V_risk, np.shape(V_risk_C)[1])).reshape((np.shape(V_risk_C)[0], np.shape(V_risk_C)[1]))
    #P[G, :] = 1
    #V_risk_C[:, :] = V_risk.T
    #P[:, :] = P_risk.T
    P = (np.repeat(P_risk, np.shape(P)[1])).reshape((np.shape(P)[0], np.shape(P)[1]))
    P[G, :] = 1
    #pi[:, :] = pi_risk.T
    pi = (np.repeat(pi_risk, np.shape(pi)[1])).reshape((np.shape(pi)[0], np.shape(pi)[1]))

    V_a = {A[i]: i for i in range(n_actions)}
    Q = np.zeros((len(S), C_max + c + 1, n_actions))
    QP = np.zeros((len(S), C_max + c + 1, n_actions))

    alpha = 1
    epsilon = 1
    for e in range(n_episodes):
        done = False
        current_state, C = getInitialState(initialState, len(S), C_max)
        alpha = get_alpha(e, min_alpha, n_episodes, alpha)
        epsilon = get_epsilon(e, min_epsilon, n_episodes, epsilon)

        #C = 0
        while not done:
            if C > C_max:
                break
            action = choose_actionQGubs(A, V_i[current_state], C, epsilon, Q)
            new_state, reward, done = mdp.step(mdp_obj, current_state, action)
            update_qegubs(V, V_risk_C, P, pi, Q, QP, V_i[current_state], C, V_a[action], reward, V_i[new_state], alpha, A, lamb, k_g, S, initialState, epsilon, e, ReplayX, Replay)
            C += (reward * -1)
            current_state = new_state

    print(A)
    return V, P, pi, ReplayX