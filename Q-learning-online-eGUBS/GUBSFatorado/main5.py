import utils
import numpy as np
from gubs import Qgubs

def try_int(key):
    try:
        return int(key)
    except:
        return key

args = utils.parse_argsQGubs()
mdp = utils.read_json(args.file_input)
l = args.lamb
def u(c): return np.exp(-l * c)

S = sorted(mdp.keys(), key=try_int)
V_i = {S[i]: i for i in range(len(S))}
V, pi = Qgubs(mdp,S,V_i,args.n_episodes, args.min_alpha, args.min_epsilon, args.gamma, args.initialState, args.c_max, u, args.kg)
print('V: ', V)
print('pi: ', pi)