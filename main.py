import inspect
import os
import numpy as np
import scipy
from random import choice

from kaggle_environments import evaluate, make, utils

env = make('connectx', debug=True)


# env.render()


# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    cfg_rows = configuration.rows
    cfg_cols = configuration.columns
    cfg_inarow = configuration.inarow
    priorities = [3, 4, 2, 5, 1, 0, 6]

    def select_first(observation):
        for i in priorities:
            if observation.board[i] == 0:
                return i

    def winning_next(observation, player_num, cfg_rows, cfg_cols, cfg_inarow):
        """
        Try to find the next winning step for a specific player.
        ---------
        observation : json
            Observation of the board.
        player_num : int
            Player number (self = 1, enemy  = 2).
        Returns
        -------
        integer
            If a winning step is possible, returns the column. If not, returns -99.
        """

        # Check horizontal possibility:
        for i in reversed(range(cfg_rows)):
            for j in reversed(range(cfg_cols - cfg_inarow)):
                if observation.board[j + cfg_cols * i] == player_num \
                        and observation.board[j + cfg_cols * i + 1] == player_num \
                        and observation.board[j + cfg_cols * i + 2] == player_num:
                    # Test if right bounded (position 4, 11, 18, 25, 32, 39):
                    if j != (cfg_cols - cfg_inarow + 1) + cfg_cols * i and observation.board[j + cfg_cols * i + 3] == 0:
                        return j + 3
                    # Test if left bounded (position 0, 7, 14, 21, 28, 35)
                    elif j != cfg_cols * i and observation.board[j + cfg_cols * i - 1] == 0:
                        return j - 1

        # Check vertical possibility:
        for i in range(cfg_cols):
            for j in reversed(range(cfg_rows - cfg_inarow)):
                if observation.board[i + cfg_cols * (j + 1)] == player_num \
                        and observation.board[i + cfg_cols * (j + 2)] == player_num \
                        and observation.board[i + cfg_cols * (j + 3)] == player_num \
                        and observation.board[i + cfg_cols * (j)] == 0:
                    return i

        # Check diagonal (top to right bottom) possibility:
        for i in range(cfg_rows - cfg_inarow - 4):
            for j in range(cfg_cols - cfg_inarow - 1):
                if observation.board[j + cfg_cols * i] == player_num \
                        and observation.board[j + 1 + cfg_cols * (i + 1)] == player_num \
                        and observation.board[j + 2 + cfg_cols * (i + 2)] == player_num:
                    # Test if right bounded (position 4, 11, 18, 25, 32, 39)
                    # and bottom bounded (35, 36, 37, 38, 39, 40, 41):
                    if j != (cfg_cols - cfg_inarow + 1) + cfg_cols * i \
                            and j not in range(cfg_cols * (cfg_rows - 1), cfg_cols * cfg_rows) \
                            and observation.board[j + 3 + cfg_cols * (i + 3)] == 0:
                        return j + 3
                    # Test if left bounded (position 0, 7, 14, 21, 28, 35) and top bounded (0, 1, 2, 3, 4, 5, 6)
                    elif j != cfg_cols * i and j not in range(cfg_cols) and observation.board[
                        j - 1 + cfg_cols * (i - 1)] == 0:
                        return j - 1

        # Check diagonal (top to left bottom) possibility:
        for i in range(cfg_rows - cfg_inarow - 1):
            for j in range(cfg_inarow - 1, cfg_cols):
                if observation.board[j + cfg_cols * i] == player_num \
                        and observation.board[j + 1 + cfg_cols * (i + 1)] == player_num \
                        and observation.board[j + 2 + cfg_cols * (i + 2)] == player_num:
                    # Test if right bounded (position 4, 11, 18, 25, 32, 39):
                    if j != (cfg_cols - cfg_inarow + 1) + cfg_cols * i \
                            and observation.board[j + 3 + cfg_cols * (i + 3)] == 0:
                        return j + 3
                    # Test if left bounded (position 0, 7, 14, 21, 28, 35) and top bounded (0, 1, 2, 3, 4, 5, 6)
                    elif j != cfg_cols * i and j not in range(cfg_cols) and observation.board[
                        j - 1 + cfg_cols * (i - 1)] == 0:
                        return j - 1
        else:
            return -99

    def round_number(observation):
        count_1 = 0
        count_2 = 0
        for i in observation.board:
            if i == 1:
                count_1 = count_1 + 1
            elif i == 2:
                count_2 = count_2 + 1
        if count_1 > count_2:
            start_player = 1
        elif count_1 < count_2:
            start_player = 2
        else:
            start_player = 0
        return count_1 + count_2, start_player  # , count_1, count_2

    r_num, start_player = round_number(observation=observation)

    # print('Round Number : {}'.format(r_num))
    if r_num < 5:
        return select_first(observation=observation)
    else:
        # Check own winning possibility:
        own_win = winning_next(observation=observation, player_num=1, cfg_rows=cfg_rows, cfg_cols=cfg_cols,
                               cfg_inarow=cfg_inarow)
        # print('Own winning: {}'.format(own_win))
        if own_win >= 0:  # is not None:
            return own_win

        # Check opposite winning possibility:
        opposite_win = winning_next(observation=observation, player_num=2, cfg_rows=cfg_rows, cfg_cols=cfg_cols,
                                    cfg_inarow=cfg_inarow)
        # print('Opposite winning: {}'.format(opposite_win))
        if opposite_win >= 0:  # is not None:
            return opposite_win
        else:
            return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    # print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)


# Run multiple episodes to estimate its performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx",
                                                        [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx",
                                                         [my_agent, "negamax"], num_episodes=10)))
'''
def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)


write_agent_to_file(my_agent, "submission.py")
'''
