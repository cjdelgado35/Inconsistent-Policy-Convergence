from datetime import datetime
import utils
import gubs
import mdp
import numpy as np

#from mdp import get_CostQgubs #river
from mdp import get_CostMonteCarloNavigator #navigator
from mdp import get_probabilitiesDE

def get_output_file_name(V, args, file_input,cmax):
    iso_date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    n_states = V.shape[0]
    file_name = f'{file_input[2:len(file_input)-5]}_{n_states}_{cmax}_{(args.lamb* 100):.0f}_{(args.kg * 100):.0f}_{iso_date}.json'

    return file_name

def get_output_file_nameMC(V, args, file_input,cmax):
    iso_date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    n_states = V.shape[0]
    file_name = f'{file_input[2:len(file_input)-5]}_{n_states}_{(args.lamb* 100):.0f}_{iso_date}.json'

    return file_name

args = utils.parse_argsLexMonteCarlo()

def automatico(args):
    mdp_obj = utils.read_json(args.file_input)
    S = sorted(mdp_obj.keys(), key=int)
    V_i = {S[i]: i for i in range(len(S))}

    V, pi, Pavg, P = gubs.LexMonteCarlo(args.lamb, mdp_obj, V_i, S, args.n_episodes, args.min_epsilon, args.initialState)


    Cust, Path, Rep = get_CostMonteCarloNavigator(V_i, pi, mdp_obj, args.initialState, S) #navigator
    #Cust, Path, Rep = get_CostQgubs(V_i, pi_gubs, mdp_obj, args.initialState, args.p, S, C_max) #river
    #P = get_probabilitiesDE(V_i, pi_gubs, S, mdp_obj, C_max) #descomentar para GUBS
    #R = [P_gubs[V_i[args.initialState],0],P[V_i[args.initialState],0],V_gubs[V_i[args.initialState],0],Cust,Rep,Path] #descomentar para GUBS
    R = [P[V_i[args.initialState]], V[V_i[args.initialState], 0], Cust, Rep, Path]
    #R = [P[V_i[args.initialState]], P[V_i[args.initialState]]]

    if args.output:
        output_filename = get_output_file_nameMC(V, args, args.file_input, 0)
        output_file_path = utils.output(
            output_filename, {'e': args.n_episodes, 'R': R, 'V': V.tolist(), 'pi': pi.tolist(), 'P': P.tolist()})
        #output_filename, {'e': args.n_episodes, 'R': R, 'V': V.tolist(), 'pi': pi.tolist(), 'P': P.tolist()})
        #output_filename, {'Cmax': C_max, 'MaxMax': MaxMax, 'V': V.tolist(), 'pi': pi.tolist(), 'P': P.tolist()})
        if output_file_path:
            print("Algorithm result written to ", output_file_path)

for i in range(1):
        automatico(args)
