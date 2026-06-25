import argparse
import json
import os
import numpy as np
from utils import output


def create_state_obj(adjs, heuristic, goal=False, river_type=0):
    obj = {
        'goal': goal,
        "deadend": False,
        'Adj': adjs,
        'heuristic': heuristic
    }
    return obj


def get_bank_adj(i, nx, ny, river_type=0):
    prob = 1 if river_type == 0 else 0.99
    return [
        {  # top bank state (left or right)
            'name': i,
            'A': {
                **(
                    {  # Right
                        'E': prob if river_type in (0, 1) else 1,
                        **({'N': 1 - prob} if river_type == 2 else {})
                    }
                    if int(i) % nx == 0
                    else {  # Left
                        'W': prob if river_type in (0, 1) else 1
                    }
                ),
            }
        },
        *([{  # Left bank
            'name': str(int(i) + 1),
            'A': {'E': 1} if river_type == 0 else {
                'N': 1 - prob,
                'S': 1 - prob,
                'E': 1,
                'W': 1 - prob
            } if river_type == 1 else {
                'N': 1 - prob,
                'E': 1
            }
        }] if int(i) % nx != 0 else []),
        *([{  # Right bank
            'name': str(int(i) - 1),
            'A': {'W': 1} if river_type in (0, 2) else {
                'N': 1 - prob,
                'S': 1 - prob,
                'E': 1 - prob,
                'W': 1
            }
        }] if int(i) % nx == 0 else []),
        {  # State at north
            'name': str(int(i) - nx),
            'A': {'N': prob}
        },
        {  # State at south
            'name': str(int(i) + nx),
            'A': {
                'S': prob
                if river_type in (0, 1)
                else 1
            }
        }

    ]


def add_bridge_states(env, nx, ny, river_type=0):
    for i in range(1, nx + 1):
        env[str(i)] = {
            'goal': False,
            "deadend": False,
            'heuristic': ny + (nx - i) - 1,
            'Adj': [
                {
                    'name': str(i),
                    'A': {
                        'N': 0.99 if river_type == 2 and i != nx else 1,
                        **({'W': 1} if i == 1 else {}),
                        ** ({'E': 1} if i == nx else {})
                    }
                },
                {
                    'name': str(i + nx),
                    'A': {'S': 1}
                },
                *([{
                    'name': str(i + 1),
                    'A': {'E': 1, **({'N': 0.01} if river_type == 2 else {})}
                }] if i != nx else []),
                *([{
                    'name': str(i - 1),
                    'A': {'W': 1}
                }] if i != 1 else []),
            ]
        }
    return env


def add_bank_states(env, nx, ny, river_type=0):
    for i in range(1, ny):
        if i < ny - 1:
            cell_i_1 = str(i * nx + 1)
            h1 = (ny - (int(cell_i_1) // nx)) + nx - 2
            env[cell_i_1] = create_state_obj(
                get_bank_adj(cell_i_1, nx, ny, river_type), h1)
        cell_i_2 = str((i + 1) * nx)
        h2 = ny - (int(cell_i_2) // nx)
        env[cell_i_2] = create_state_obj(
            get_bank_adj(cell_i_2, nx, ny, river_type), h2)

    goal_state_i = str(nx * ny)
    env[goal_state_i] = create_state_obj([
        {
            'name': goal_state_i,
            'A': {
                'N': 1,
                'S': 1,
                'E': 1,
                'W': 1,
            }
        }
    ], 0, goal=True)

    return env


def add_river_states(env, nx, ny, p, river_type=0):
    river_prob = p if river_type == 0 else p * p
    success_prob = 1 - p if river_type == 0 else (1 - p) * (1 - p)
    stays_prob = 1 if river_type == 0 else 2 * p * (1 - p)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            index = i * nx + j + 1
            env[str(index)] = {
                'goal': False,
                "deadend": False,
                'heuristic': nx - i + ny - j - 2,
                'Adj': [
                    *([] if river_type == 0 else [{
                        'name': str(index),
                        'A': {
                            'N': stays_prob,
                            **({} if river_type == 2 else {'S': stays_prob}),
                            'E': stays_prob / 2,
                            'W': stays_prob / 2
                        }
                    }]),
                    {
                        # goes down the river
                        'name': str(index + nx),
                        'A': {
                            'N': river_prob,
                            'S': 1,
                            'E': river_prob,
                            'W': river_prob
                        }
                    },
                    *([{
                        # goes down the river and to the right
                        'name': str(index + nx + 1),
                        'A': {'E': stays_prob / 2}
                    }] if river_type == 2 else []),
                    *([{
                        # goes down the river and to the left
                        'name': str(index + nx - 1),
                        'A': {'W': stays_prob / 2}
                    }] if river_type == 2 else []),
                    {
                        'name': str(index - nx),
                        'A': {'N': success_prob}
                    },
                    {
                        'name': str(index + 1),
                        'A': {'E': success_prob}
                    },
                    {
                        'name': str(index - 1),
                        'A': {'W': success_prob}
                    }
                ]
            }
    return env


def add_waterfall_states(env, nx, ny, river_type=0):
    # começa no 1 + ny * nx
    begin = 1 + (ny - 1) * nx
    end = nx * ny - 1
    for i in range(begin, end + 1):
        env[str(i)] = {
            'goal': False,
            "deadend": True,
            'heuristic': end - i + 1,
            'Adj': [{
                'name': str(i),
                'A': {
                    'N': 1,
                    'S': 1,
                    'E': 1,
                    'W': 1,
                }
            }]
        }
    return env


def create_env(nx, ny, p, river_type=0):
    env = {}
    add_bridge_states(env, nx, ny, river_type)
    add_bank_states(env, nx, ny, river_type)
    add_river_states(env, nx, ny, p, river_type)
    add_waterfall_states(env, nx, ny)
    return env


DEFAULT_P = 0.6
DEFAULT_NX = 5
DEFAULT_NY = 20
DEFAULT_DEST_DIR = '.'
DEFAULT_TYPE = 0

parser = argparse.ArgumentParser(
    description='River problem generator'
)
parser.add_argument('-p', dest='p', default=DEFAULT_P, type=float)
parser.add_argument('--nx', dest='nx', default=DEFAULT_NX, type=int)
parser.add_argument('--ny', dest='ny', default=DEFAULT_NY, type=int)
parser.add_argument('--dest_dir', dest='dest_dir', default=DEFAULT_DEST_DIR)
parser.add_argument('--type', dest='type',
                    choices=['0', '1', '2'], default=DEFAULT_TYPE)

args = parser.parse_args()
nx = args.nx
ny = args.ny
p = args.p
river_type = int(args.type)

env = create_env(nx, ny, p, river_type)

output('river%d-%d-%d-%d.json' %
       (nx, ny, p * 100, river_type), env, args.dest_dir)
