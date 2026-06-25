import utils
from gubs import Qlearning

def try_int(key):
    try:
        return int(key)
    except:
        return key

args = utils.parse_argsQLearning()
mdp = utils.read_json(args.file_input)

S = sorted(mdp.keys(), key=try_int)
V_i = {S[i]: i for i in range(len(S))}
Qlearning(mdp,S,V_i,args.n_episodes, args.min_alpha, args.min_epsilon, args.gamma, args.initialState)