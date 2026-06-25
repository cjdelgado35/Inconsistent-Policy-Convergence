import os
import argparse
import json


def read_json(file_name):
    with open(file_name) as json_data:
        return json.load(json_data)


def try_int(key):
    try:
        return int(key)
    except:
        return key


DEFAULT_FILE_INPUT = './navigator4-10-0-0.json'#'./navigator3-15-0-0.json''./navigator4-10-0-0.json'#'./navigator3-15-0-0.json'#'./navigator4-10-0-0.json'#'./navigator3-10-0-0.json'#'./river5-20-40-0.json'#'./river7-10-40-0.json'#'./river7-10-40-0.json'#'./river3-8-50-0.json'#'./testestochastic.json'#'./teste.json' #'./testedeadend.json' #'./env1.json'
DEFAULT_WIDTH = 4
DEFAULT_P = 0 #Prob to dead end
DEFAULT_KG = 0
DEFAULT_LAMBDA = -0.2 #Para QeGubs:-0.2 #Para QGubs: 0.1
DEFAULT_OUTPUT = True
DEFAULT_OUTPUT_DIR = "./outpute"
NEPISODES = 1000000 # 2000000# training episodes
NEPISODESe =2000000 # training episodes
MINALPHA = 0.000001#0.000001 # learning rate
MINEPSILON = 0.1#0.4#0.1 # exploration rate
GAMMA = 1 # discount factor
INITIALSTATE = '37'#'118'#'43'#'37'#'28'#'91' # Initial state


def output(output_filename, data, output_dir=DEFAULT_OUTPUT_DIR):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, output_filename)

    with open(output_file_path, 'w') as fp:
        json.dump(data, fp, indent=2)

    return output_file_path


def parse_args():

    parser = argparse.ArgumentParser(
        description='GUBS (Goals with Utility-Based Semantic) algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--write_output', dest='output',
                        default=DEFAULT_OUTPUT,
                        action="store_true",
                        help="Defines whether or not to write the algorithm output to a file (default: %s)" % DEFAULT_OUTPUT)
    parser.add_argument('--c_max', dest='c_max', required=True,
                        type=int,
                        help="Maximum cost used in the algorithm")
    parser.add_argument('--kg', dest='kg',
                        default=DEFAULT_KG,
                        type=float,
                        help="Kg costant cost on goal states (default: %s)" % DEFAULT_KG)
    parser.add_argument('--lambda', dest='lamb',
                        default=DEFAULT_LAMBDA,
                        type=float,
                        help="Lambda risk factor (default: %s)" % DEFAULT_LAMBDA)
    return parser.parse_args()

def parse_argsVI():

    parser = argparse.ArgumentParser(
        description='VI(Value Iteration) algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--write_output', dest='output',
                        default=DEFAULT_OUTPUT,
                        action="store_true",
                        help="Defines whether or not to write the algorithm output to a file (default: %s)" % DEFAULT_OUTPUT)
    parser.add_argument('--lambda', dest='lamb',
                        default=DEFAULT_LAMBDA,
                        type=float,
                        help="Lambda risk factor (default: %s)" % DEFAULT_LAMBDA)
    return parser.parse_args()

def parse_argsQLearning():

    parser = argparse.ArgumentParser(
        description='QLearning algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--write_output', dest='output',
                        default=DEFAULT_OUTPUT,
                        action="store_true",
                        help="Defines whether or not to write the algorithm output to a file (default: %s)" % DEFAULT_OUTPUT)
    parser.add_argument('--n_episodes', dest='n_episodes',
                        default=NEPISODES,
                        type=int,
                        help="Maximum number of episodes")
    parser.add_argument('--min_alpha', dest='min_alpha',
                        default=MINALPHA,
                        type=int,
                        help="Alpha QLearning")
    parser.add_argument('--min_epsilon', dest='min_epsilon',
                        default=MINEPSILON,
                        type=int,
                        help="Epsilon QLearning")
    parser.add_argument('--gamma', dest='gamma',
                        default=GAMMA,
                        type=int,
                        help="Epsilon QLearning")
    parser.add_argument('--initialState', dest='initialState',
                        default=INITIALSTATE,
                        type=str,
                        help="InitialState QLearning")
    return parser.parse_args()

def parse_argsQGubs():

    parser = argparse.ArgumentParser(
        description='QGubs algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--write_output', dest='output',
                        default=DEFAULT_OUTPUT,
                        action="store_true",
                        help="Defines whether or not to write the algorithm output to a file (default: %s)" % DEFAULT_OUTPUT)
    parser.add_argument('--n_episodes', dest='n_episodes',
                        default=NEPISODES,
                        type=int,
                        help="Maximum number of episodes")
    parser.add_argument('--min_alpha', dest='min_alpha',
                        default=MINALPHA,
                        type=int,
                        help="Alpha QLearning")
    parser.add_argument('--min_epsilon', dest='min_epsilon',
                        default=MINEPSILON,
                        type=int,
                        help="Epsilon QLearning")
    parser.add_argument('--gamma', dest='gamma',
                        default=GAMMA,
                        type=int,
                        help="Epsilon QLearning")
    parser.add_argument('--initialState', dest='initialState',
                        default=INITIALSTATE,
                        type=str,
                        help="InitialState QLearning")
    parser.add_argument('--c_max', dest='c_max', required=True,
                        type=int,
                        help="Maximum cost used in the algorithm")
    parser.add_argument('--kg', dest='kg',
                        default=DEFAULT_KG,
                        type=float,
                        help="Kg costant cost on goal states (default: %s)" % DEFAULT_KG)
    parser.add_argument('--lambda', dest='lamb',
                        default=DEFAULT_LAMBDA,
                        type=float,
                        help="Lambda risk factor (default: %s)" % DEFAULT_LAMBDA)
    return parser.parse_args()
def parse_argsQeGubs():

    parser = argparse.ArgumentParser(
        description='QeGubs algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--write_output', dest='output',
                        default=DEFAULT_OUTPUT,
                        action="store_true",
                        help="Defines whether or not to write the algorithm output to a file (default: %s)" % DEFAULT_OUTPUT)
    parser.add_argument('--n_episodes', dest='n_episodes',
                        default=NEPISODES,
                        type=int,
                        help="Maximum number of episodes")
    parser.add_argument('--n_episodese', dest='n_episodese',
                        default=NEPISODESe,
                        type=int,
                        help="Maximum number of episodes")
    parser.add_argument('--min_alpha', dest='min_alpha',
                        default=MINALPHA,
                        type=int,
                        help="Alpha QLearning")
    parser.add_argument('--min_epsilon', dest='min_epsilon',
                        default=MINEPSILON,
                        type=int,
                        help="Epsilon QLearning")
    parser.add_argument('--gamma', dest='gamma',
                        default=GAMMA,
                        type=int,
                        help="Epsilon QLearning")
    parser.add_argument('--initialState', dest='initialState',
                        default=INITIALSTATE,
                        type=str,
                        help="InitialState QLearning")
    parser.add_argument('--kg', dest='kg',
                        default=DEFAULT_KG,
                        type=float,
                        help="Kg costant cost on goal states (default: %s)" % DEFAULT_KG)
    parser.add_argument('--lambda', dest='lamb',
                        default=DEFAULT_LAMBDA,
                        type=float,
                        help="Lambda risk factor (default: %s)" % DEFAULT_LAMBDA)
    parser.add_argument('--width', dest='width',
                        default=DEFAULT_WIDTH,
                        type=int,
                        help="river width")
    parser.add_argument('--p', dest='p',
                        default=DEFAULT_P,
                        type=float,
                        help="Probability to deadend (default: %s)" % DEFAULT_P)
    return parser.parse_args()

def parse_argsLexMonteCarlo():

    parser = argparse.ArgumentParser(
        description='Monte Carlo Lexicographic algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--write_output', dest='output',
                        default=DEFAULT_OUTPUT,
                        action="store_true",
                        help="Defines whether or not to write the algorithm output to a file (default: %s)" % DEFAULT_OUTPUT)
    parser.add_argument('--n_episodes', dest='n_episodes',
                        default=NEPISODES,
                        type=int,
                        help="Maximum number of episodes")
    parser.add_argument('--n_episodese', dest='n_episodese',
                        default=NEPISODESe,
                        type=int,
                        help="Maximum number of episodes")
    parser.add_argument('--min_epsilon', dest='min_epsilon',
                        default=MINEPSILON,
                        type=int,
                        help="Epsilon QLearning")
    parser.add_argument('--initialState', dest='initialState',
                        default=INITIALSTATE,
                        type=str,
                        help="InitialState QLearning")
    parser.add_argument('--lambda', dest='lamb',
                        default=DEFAULT_LAMBDA,
                        type=float,
                        help="Lambda risk factor (default: %s)" % DEFAULT_LAMBDA)
    parser.add_argument('--width', dest='width',
                        default=DEFAULT_WIDTH,
                        type=int,
                        help="river width")
    parser.add_argument('--p', dest='p',
                        default=DEFAULT_P,
                        type=float,
                        help="Probability to deadend (default: %s)" % DEFAULT_P)
    return parser.parse_args()
