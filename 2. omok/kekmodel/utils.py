__all__ = ["valid_actions", "check_win", "update_state",
           "render_str", "get_state_tf", "get_state_pt", "get_action"]
import numpy as np


def valid_actions(game_board):
    actions = []
    count = 0
    state_size = len(game_board)

    for i in range(state_size):
        for j in range(state_size):
            if game_board[i][j] == 0:
                actions.append([(i, j), count])
            count += 1

    return actions


# Check win
def check_win(game_board, win_mark):
    num_mark = np.count_nonzero(game_board)
    state_size = len(game_board)

    current_grid = np.zeros([win_mark, win_mark])

    # check win
    for row in range(state_size - win_mark + 1):
        for col in range(state_size - win_mark + 1):
            current_grid = game_board[row: row + win_mark, col: col + win_mark]

            sum_horizontal = np.sum(current_grid, axis=1)             # hotizontal
            sum_vertical = np.sum(current_grid, axis=0)             # vertical
            sum_diagonal_1 = np.sum(current_grid.diagonal())            # diagonal -> lower right
            sum_diagonal_2 = np.sum(np.flipud(current_grid).diagonal())  # diagonal -> upper right

            # Black wins! (Horizontal and Vertical)
            if win_mark in sum_horizontal or win_mark in sum_vertical:
                return 1

            # Black wins! (Diagonal)
            if win_mark == sum_diagonal_1 or win_mark == sum_diagonal_2:
                return 1

            # White wins! (Horizontal and Vertical)
            if -win_mark in sum_horizontal or -win_mark in sum_vertical:
                return 2

            # White wins! (Diagonal)
            if -win_mark == sum_diagonal_1 or -win_mark == sum_diagonal_2:
                return 2

    # Draw (board is full)
    if num_mark == state_size * state_size:
        return 3

    # If No winner or no draw
    return 0


def update_state(state, turn, x_idx, y_idx):
    state[:, :, 1:16] = state[:, :, 0:15]
    state[:, :, 0] = state[:, :, 2]
    state[y_idx, x_idx, 0] = 1
    state[:, :, 16] = turn
    state = np.int8(state)

    return state


def render_str(gameboard, GAMEBOARD_SIZE):
    count = np.count_nonzero(gameboard)
    board_str = '\n  0 1 2 3 4 5 6 7 8\n'
    for i in range(GAMEBOARD_SIZE):
        for j in range(GAMEBOARD_SIZE):
            if j == 0:
                board_str += '{}'.format(i)
            if gameboard[i][j] == 0:
                board_str += ' .'
            if gameboard[i][j] == 1:
                board_str += ' O'
            if gameboard[i][j] == -1:
                board_str += ' X'
            if j == GAMEBOARD_SIZE - 1:
                board_str += ' \n'
        if i == GAMEBOARD_SIZE - 1:
            board_str += '  ***  MOVE: {} ***'.format(count)
    print(board_str)


def get_state_tf(id, turn, state_size, channel_size):
    state = np.zeros([state_size, state_size, channel_size])
    length_game = len(id)

    state_1 = np.zeros([state_size, state_size])
    state_2 = np.zeros([state_size, state_size])

    channel_idx = channel_size - 1

    for i in range(length_game):
        row_idx = int(id[i] / state_size)
        col_idx = int(id[i] % state_size)

        if i != 0:
            if i % 2 == 0:
                state_1[row_idx, col_idx] = 1
            else:
                state_2[row_idx, col_idx] = 1

        if length_game - i < channel_size:
            channel_idx = length_game - i - 1

            if i % 2 == 0:
                state[:, :, channel_idx] = state_1
            else:
                state[:, :, channel_idx] = state_2

    if turn == 0:
        state[:, :, channel_size - 1] = 1
    else:
        state[:, :, channel_size - 1] = 0

    return state


def get_state_pt(id, turn, state_size, channel_size):
    state = np.zeros([channel_size, state_size, state_size], 'float')
    length_game = len(id)

    state_1 = np.zeros([state_size, state_size], 'float')
    state_2 = np.zeros([state_size, state_size], 'float')

    channel_idx = channel_size - 1

    for i in range(length_game):
        row_idx = int(id[i] / state_size)
        col_idx = int(id[i] % state_size)

        if i != 0:
            if i % 2 == 0:
                state_1[row_idx, col_idx] = 1
            else:
                state_2[row_idx, col_idx] = 1

        if length_game - i < channel_size:
            channel_idx = length_game - i - 1

            if i % 2 == 0:
                state[channel_idx] = state_1
            else:
                state[channel_idx] = state_2

    if turn == 0:
        state[channel_size - 1] = 1
    else:
        state[channel_size - 1] = 0

    return state


def get_action(pi, tau):
    action_size = len(pi)
    action = np.zeros(action_size)
    if tau == 0:
        actions = np.argwhere(pi == pi.max()).flatten()
        action_index = actions[np.random.choice(len(actions))]
    else:
        action_index = np.random.choice(action_size, p=pi)
    action[action_index] = 1
    return action, action_index
