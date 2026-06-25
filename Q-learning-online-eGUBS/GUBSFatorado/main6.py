from datetime import datetime
import utils
import gubs
import mdp
import numpy as np

#from mdp import get_CostQgubs #river
from mdp import get_CostQgubsNavigator #navigator
from mdp import get_probabilitiesDE

def get_output_file_name(V, args, file_input,cmax):
    iso_date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    n_states = V.shape[0]
    file_name = f'{file_input[2:len(file_input)-5]}_{n_states}_{cmax}_{(args.lamb* 100):.0f}_{(args.kg * 100):.0f}_{iso_date}.json'

    return file_name

args = utils.parse_argsQeGubs()

def automatico(i, kg, args):
    args.kg = kg
    mdp_obj = utils.read_json(args.file_input)
    S = sorted(mdp_obj.keys(), key=int)
    A = mdp.get_actions(mdp_obj)
    V_i = {S[i]: i for i in range(len(S))}

    V_gubs, P_gubs, pi_gubs, C_max, V, pi, P, allcmax, allcmax2, stateCmax, actionCmax = gubs.eGubsQlearning(args.width, args.lamb, mdp_obj, V_i, S, args.n_episodes, args.n_episodese, args.min_alpha, args.min_epsilon, args.initialState,args.kg)
    #MaxMax = np.max(Max)#Solo para prueba

    Cust, Path, Rep = get_CostQgubsNavigator(V_i, pi_gubs, mdp_obj, args.initialState, S, C_max) #navigator
    #Cust, Path, Rep = get_CostQgubs(V_i, pi_gubs, mdp_obj, args.initialState, args.p, S, C_max) #river
    #P = get_probabilitiesDE(V_i, pi_gubs, S, mdp_obj, C_max) #descomentar para GUBS
    #R = [P_gubs[V_i[args.initialState],0],P[V_i[args.initialState],0],V_gubs[V_i[args.initialState],0],Cust,Rep,Path] #descomentar para GUBS
    R = [P_gubs[V_i[args.initialState], 0], P[V_i[args.initialState]], V_gubs[V_i[args.initialState], 0], Cust, Rep, Path]
    #R = [P_gubs[V_i[args.initialState], 0], P[V_i[args.initialState]], V_gubs[V_i[args.initialState], 0]]
    if args.output:
        output_filename = get_output_file_name(V_gubs, args, args.file_input, C_max)
        output_file_path = utils.output(
        output_filename, {'e': args.n_episodes, 'Cmax': C_max, 'R': R, 'V': V.tolist(), 'pi': pi.tolist(), 'P': P.tolist()})
        #output_filename, {'Cmax': C_max, 'MaxMax': MaxMax, 'V': V.tolist(), 'pi': pi.tolist(), 'P': P.tolist()})
        if output_file_path:
            print("Algorithm result written to ", output_file_path)
        utils.output('cmax.json',{'cmax': allcmax})
        #utils.output('cmaxreal.json', {'stateCmax': int(stateCmax), 'actionCmax': int(actionCmax), 'cmax': allcmax2})
        utils.output('cmaxreal.json', {'cmax': allcmax2})

L = [0.01,0.05,0.1,0.2,0.3,1]#navigation
#L = [0.2]#navigation
#L = [0.01,0.2,0.5,1,5,16]#rio

for i in range(5):
    for e in L:
        automatico(i, e, args)
