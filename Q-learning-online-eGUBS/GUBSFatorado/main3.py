import utils
from gubs import SPUDD
from mdp import get_probabilitiesVI

def try_int(key):
    try:
        return int(key)
    except:
        return key

args = utils.parse_argsVI()
mdp = utils.read_json(args.file_input)

S = sorted(mdp.keys(), key=try_int)
V_i = {S[i]: i for i in range(len(S))}
SPUDD(mdp,S,V_i)
#V_i = {S[i]: i for i in range(len(S))}
#V, pi, itera = vi(mdp, V_i, S)
#P = get_probabilitiesVI(V_i, pi, S, mdp, 1e-3)
