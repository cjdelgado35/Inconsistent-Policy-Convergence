import numpy as np


def flatten(l):
    return [x for l_i in l for x in l_i]


def get_actions(mdp):
    adjs = map(lambda s: s['Adj'], mdp.values())
    actions = map(lambda s: list(s['A'].keys()), flatten(adjs))
    return list(set(flatten(actions)))

def find_reachable(s, a, mdp):
    """ Find states that are reachable from state 's' after executing action 'a' """
    all_reachable_from_s = mdp[s]['Adj']
    return list(filter(
        lambda obj_s_: a in obj_s_['A'],
        all_reachable_from_s
    ))


def Q(s, C, a, u, V, V_i, mdp, c=1):
    reachable = find_reachable(s, a, mdp)
    c_ = 0 if mdp[s]['goal'] else c
    s_a_cost = u(C + c_) - u(C)
    #print(' ', C, s, a, s_a_cost, sum([
    #    V[V_i[s_['name']], C + c_] * s_['A'][a] for s_ in reachable]), s_a_cost + sum([
    #        V[V_i[s_['name']], C + c_] * s_['A'][a] for s_ in reachable]))
    return s_a_cost + sum([
        V[V_i[s_['name']], C + c_] * s_['A'][a] for s_ in reachable])

def QVI(s, a, V, V_i, mdp, c=1):
    reachable = find_reachable(s, a, mdp)
    c_ = 0 if mdp[s]['goal'] else c
    s_a_cost = c_*-1
    print(' ', s, a, s_a_cost, sum([
        V[V_i[s_['name']]] * s_['A'][a] for s_ in reachable]), s_a_cost + sum([
            V[V_i[s_['name']]] * s_['A'][a] for s_ in reachable]))
    return s_a_cost + sum([
        V[V_i[s_['name']]] * s_['A'][a] for s_ in reachable])


def get_probabilities_finite(V_i, pi, S, C_max, H, mdp, c=1):
    P = np.zeros((len(S), C_max + c + 1, H + 1))
    G = [V_i[i] for i, s in mdp.items() if s['goal']]
    P[G] = 1
    print(S)

    for h in reversed(range(H)):
        for C in reversed(range(C_max + 1)):
            for s in S:
                i_s = V_i[s]
                try:
                    a = pi[i_s, C, h]
                except IndexError:
                    a = pi[i_s, C, -1]
                reachable = find_reachable(s, a, mdp)
        #        if s == '20':
        #            print(i_s, s, a, reachable, P[V_i['20'], C + c_, h + 1])
                c_ = 0 if mdp[s]['goal'] else c
                P[V_i[s], C, h] = np.sum([s_['A'][a] * P[V_i[s_['name']], C + c_, h + 1]
                                          for s_ in reachable])
        #print(h, P[19, 0, h])
    return P


def get_avg_cost_finite(V_i, pi, S, C_max, H, mdp, c=1):
    V_cost = np.zeros((len(S), C_max + c + 1, H + 1))

    for h in reversed(range(H)):
        for C in reversed(range(C_max + 1)):
            for s in S:
                i_s = V_i[s]
                try:
                    a = pi[i_s, C, h]
                except IndexError:
                    a = pi[i_s, C, -1]
                reachable = find_reachable(s, a, mdp)
                c_ = 0 if mdp[s]['goal'] else c
                V_cost[V_i[s], C, h] = c_ + np.sum([s_['A'][a] * V_cost[V_i[s_['name']], C + c_, h + 1]
                                                    for s_ in reachable])
    return V_cost


def get_probabilities(V_i, pi, S, mdp, epsilon=1e-3):
    P = np.zeros(len(S))
    G = [V_i[i] for i, s in mdp.items() if s['goal']]
    P[G] = 1
    i = 0
    while True:
        P_ = np.array(P)
        for s in S:
            if mdp[s]['goal']:
                continue
            a = pi[V_i[s], 0]
            reachable = find_reachable(s, a, mdp)
            P_[V_i[s]] = np.sum([s_['A'][a] * P[V_i[s_['name']]]
                                 for s_ in reachable])

        if np.linalg.norm(P_ - P, np.inf) < epsilon:
            break
        P = P_
        i += 1

    return P

####CHDP

def get_probabilitiesQgubs(V_i, pi, S, mdp_obj, initialState, C_max, ReplayX, epsilon=1e-6, reward=1):
    P = np.zeros((len(S), C_max))
    G = [V_i[i] for i, s in mdp_obj.items() if s['goal']]
    P[G] = 1
    i = 0
    while True:
        D = []#estado y reward
        #Ds = []#solo estados
        Dv = []#visitados
        P_ = np.array(P)
        D.append((initialState,0))
        #Ds.append(initialState)
        while len(D) > 0:
            s = D.pop(0)
            Dv.append(s)#Ya visitado
            if mdp_obj[s[0]]['goal']:
                continue
            a = pi[V_i[s[0]], s[1]]
            reachable = find_reachable(s[0], a, mdp_obj)
            P_[V_i[s[0]],s[1]] = np.sum([s_['A'][a] * P[V_i[s_['name']],s[1]+reward]
                                 for s_ in reachable])

            #if valor > P_[V_i[s[0]],0]:#Máximo ya que es el más convergido, la prob es la misma independiente de c
            #    P_[V_i[s[0]]] = valor
            for e in reachable:
                if s[1]+reward < C_max - 1:
                    if (e['name'],s[1]+reward) not in D and (e['name'],s[1]+reward) not in Dv and ReplayX[V_i[e['name']],s[1]+reward] > 0:#no repetir estado
                    # si es un estado con política no actualiza, no incluirlo, prob 0.
                        D.append((e['name'],s[1]+reward))
                        #Ds.append(e['name'])
        if np.linalg.norm(P_ - P, np.inf) < epsilon:
            break
        P = P_
        i += 1
    return P

def get_CostQgubs(V_i, pi, mdp_obj, initialState, p, S, cmax):#Solo poblema del rio
    Path = []
    i = 0
    s = initialState
    while not mdp_obj[s]['goal'] and i < cmax:
        Path.append(s)
        a = pi[V_i[s],i]
        reachable = find_reachable(s, a, mdp_obj)
        states = [(s_['name'],s_['A'][a]) for s_ in reachable]
        for e in states:
            if not e[1] == p:
                s = e[0]
                break
        i += 1
    #Calculo de Repetidos
    Rep = np.zeros(len(S))
    for e in Path:
        Rep[int(e)] += 1
    return len(Path),Path,np.mean(Rep[np.nonzero(Rep)])

def get_CostQgubsNavigator(V_i, pi, mdp_obj, initialState, S, cmax):#Solo Navigator
    deadend = [i for i, s in mdp_obj.items() if s['deadend']]
    Path = []
    i = 0
    s = initialState
    while not mdp_obj[s]['goal'] and i < cmax:
        Path.append(s)
        a = pi[V_i[s],i]
        reachable = find_reachable(s, a, mdp_obj)
        states = [(s_['name'],s_['A'][a]) for s_ in reachable]
        for e in states:
            if not e[0] == deadend[0]:
                s = e[0]
                break
        i += 1
    # Calculo de Repetidos
    Rep = np.zeros(len(S))
    for e in Path:
        Rep[int(e)] += 1
    return len(Path),Path,np.mean(Rep[np.nonzero(Rep)])

def get_CostMonteCarloNavigator(V_i, pi, mdp_obj, initialState, S):#Solo Navigator
    deadend = [i for i, s in mdp_obj.items() if s['deadend']]
    Path = []
    i = 0
    s = initialState
    while not mdp_obj[s]['goal']:
        Path.append(s)
        a = pi[V_i[s]]
        reachable = find_reachable(s, a, mdp_obj)
        states = [(s_['name'],s_['A'][a]) for s_ in reachable]
        for e in states:
            if not e[0] == deadend[0]:
                s = e[0]
                break
        i += 1
    # Calculo de Repetidos
    Rep = np.zeros(len(S))
    for e in Path:
        Rep[int(e)] += 1
    return len(Path),Path,np.mean(Rep[np.nonzero(Rep)])

def get_probabilitiesVI(V_i, pi, S, mdp, epsilon=1e-3):
    P = np.zeros(len(S))
    G = [V_i[i] for i, s in mdp.items() if s['goal']]
    P[G] = 1
    i = 0
    while True:
        P_ = np.array(P)
        for s in S:
            if mdp[s]['goal']:
                continue
            a = pi[V_i[s]]
            reachable = find_reachable(s, a, mdp)
            P_[V_i[s]] = np.sum([s_['A'][a] * P[V_i[s_['name']]]
                                 for s_ in reachable])

        if np.linalg.norm(P_ - P, np.inf) < epsilon:
            break
        P = P_
        i += 1

    return P

def find_parents(mdp, S, V_i, A, A_i, parents):
    """ Find states that are parents from state 's' after executing each action 'a' """
    for s in S:
        for a in A:
            all_reachable_from_s = mdp[s]['Adj']
            L = list(filter(
                lambda obj_s_: a in obj_s_['A'],
                all_reachable_from_s
            ))
            if len(L)>0:
                for state in L:
                    parents[A_i[a], V_i[state['name']]].append((s, state['A'][a]))

## Q Learning
def step(mdp_obj, current_state, action):
    all_reachable_from_s = mdp_obj[current_state]['Adj']
    L = list(filter(
        lambda obj_s_: action in obj_s_['A'],
        all_reachable_from_s
    ))
    P = []
    states = []
    for i in range(len(L)):
        P.append(L[i]["A"][action])
        states.append(L[i]["name"])
    new_state = states[np.random.choice(np.arange(0, len(states)), p=P)]
    reward = -1
    goal = mdp_obj[current_state]['goal']
    deadend = mdp_obj[current_state]['deadend']
    done = False
    if goal or deadend:
        done = True
        if goal:
            reward = 0
    return new_state, reward, done

def stepMC(mdp_obj, current_state, action):
    all_reachable_from_s = mdp_obj[current_state]['Adj']
    L = list(filter(
        lambda obj_s_: action in obj_s_['A'],
        all_reachable_from_s
    ))
    P = []
    states = []
    for i in range(len(L)):
        P.append(L[i]["A"][action])
        states.append(L[i]["name"])
    new_state = states[np.random.choice(np.arange(0, len(states)), p=P)]
    reward = 0
    cost = 1
    goal = mdp_obj[current_state]['goal']
    deadend = mdp_obj[current_state]['deadend']
    done = False
    if goal or deadend:
        done = True
        if goal:
            reward = 1
        if deadend:
            cost = -1
    return new_state, reward, done, cost

def get_probabilitiesDE(V_i, pi, S, mdp, C_max, epsilon=1e-3):#Get probabilities with dead ends
    P = np.zeros((len(S), C_max+1))
    G = [V_i[i] for i, s in mdp.items() if s['goal']]
    D = [V_i[i] for i, s in mdp.items() if s['deadend']]
    P[G] = 1
    P[D] = 0
    i = 0
    while True:
        P_ = np.array(P)
        for s in S:
            if mdp[s]['goal'] or mdp[s]['deadend']:
                continue
            for c in range(0, C_max):
                a = pi[V_i[s], c]
                reachable = find_reachable(s, a, mdp)
                P_[V_i[s], c] = np.sum([s_['A'][a] * P[V_i[s_['name']], c+1]
                                     for s_ in reachable])
        if np.linalg.norm(P_ - P, np.inf) < epsilon:
            break
        P = P_
        i += 1

    return P

def render(Q, nstates, A, nactions, x):
    print(A)
    res = ""
    for s in range(nstates):
        res = res + "["
        for a in range(nactions):
            res = res + str("{:.5f}".format(Q[s, a])) + " "
        res = res + "]"
        if (s + 1) % x == 0:
            print(res)
            res = ""
