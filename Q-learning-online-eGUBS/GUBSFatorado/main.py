from datetime import datetime
import numpy as np
import utils
from gubs import gubs
from mdp import get_probabilities


def try_int(key):
    try:
        return int(key)
    except:
        return key


def get_output_file_name(V, args):
    iso_date = datetime.isoformat(datetime.today())
    n_states = V.shape[0]
    file_name = f'{n_states}_{args.c_max}_{(args.lamb* 100):.0f}_{(args.kg * 100):.0f}_{iso_date}.json'

    return file_name


args = utils.parse_args()
c_max = args.c_max
l = args.lamb
kg = args.kg
mdp = utils.read_json(args.file_input)


def u(c): return np.exp(-l * c)


S = sorted(mdp.keys(), key=try_int)
V_i = {S[i]: i for i in range(len(S))}
V, pi, itera, VA = gubs(c_max, u, kg, mdp, V_i, S)
P = get_probabilities(V_i, pi, S, mdp, 1e-3)
print('V: ', V)
print('pi: ', pi)
print('P: ', P.tolist())
print('Iteraciones: ', itera)

# {
#         "name": "2",
#         "A": { "b": 1 }
#       },

#,
  # "5": {
  #   "goal": false,
  #   "heuristic": 1,
  #   "Adj": [
  #     {
  #       "name": "3",
  #       "A": { "a": 1 }
  #     }
  #   ]
  # },

if args.output:
    output_filename = get_output_file_name(V, args)
    output_file_path = utils.output(
        output_filename, {'V': V.tolist(), 'pi': pi.tolist(), 'P': P.tolist()})
    if output_file_path:
        print("Algorithm result written to ", output_file_path)
