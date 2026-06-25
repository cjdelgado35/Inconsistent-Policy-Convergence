import utils
from gubs import vi
from mdp import get_probabilitiesVI

from add import ADD

def try_int(key):
    try:
        return int(key)
    except:
        return key

args = utils.parse_argsVI()
mdp = utils.read_json(args.file_input)

S = sorted(mdp.keys(), key=try_int)
V_i = {S[i]: i for i in range(len(S))}
V, pi, itera = vi(mdp, V_i, S)
P = get_probabilitiesVI(V_i, pi, S, mdp, 1e-3)
print('V: ', V)
print('pi: ', pi)
print('P: ', P.tolist())
print('Iteraciones: ', itera)

# x1 = ADD.variable(1, 3, 4)
# print("=== x1 ===")
# print(x1)
#
# x3 = ADD.variable(3, 5, 6)
# print("=== x3 ===")
# print(x3)
#
# x3b = ADD.variable(3, 7, 8)
# print("=== x3b ===")
# print(x3b)
#
# ADD.subvariable(x1, x3, x3b)
# print("=== x1 ===")
# print(x1)

# print("=== (x1 + x2) ===")
# print(x1 + x3)

