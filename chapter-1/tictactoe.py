import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import time
from IPython import display
import pickle

state_map = [{}, {}]
win_count = {0: 0, 1: 1, -1: 0}


def init_board(init_val=-1):
    return np.full((3, 3), init_val)


def get_free_tiles(b, plyr):
    free_tiles = np.where(b == -1)
    free_ix = zip(free_tiles[0].tolist(), free_tiles[1].tolist())
    return [tuple(m) for m in free_ix]


def exploratory_move(brd, mvs, plyr, stmap):
    # return a random exploratory move from the available moves
    ix = np.random.choice(len(mvs))
    exp_mv = mvs[ix]
    brd[exp_mv] = plyr
    b_s = ":".join([str(a) for a in brd.flatten().tolist()])
    v = get_state_value(b_s, plyr, stmap)
    brd[exp_mv] = -1
    return (v, exp_mv, b_s)


def greedy_move(brd, mvs, plyr, stmap):
    # return the best possible available for the player
    state_values = []
    for mv in mvs:
        # try the move
        brd[mv] = plyr
        # get its value
        b_s = ":".join([str(a) for a in brd.flatten().tolist()])
        v = get_state_value(b_s, plyr, stmap)
        # retract the move
        brd[mv] = -1
        state_values.append((v, mv, b_s))
    # return the most fav state (greedy approach).
    return sorted(state_values, key=lambda x: -x[0])[0]


def get_state_value(s, plyr, stmap):
    # get board representation from state string
    b = np.array(list(map(int, s.split(":")))).reshape(3, 3)
    value = 0.5
    win = np.array([plyr] * 3)
    lose = np.array([abs(plyr - 1)] * 3)
    if s in stmap:
        value = stmap[s]

    # has the board filled up without win or draw?
    if np.count_nonzero(b == -1) == 0:
        value = 0

    for i in range(3):
        # did we win?
        if np.all(b[:, i] == win) or np.all(b[i, :] == win):
            value = 1
        # did we lose?
        if np.all(b[:, i] == lose) or np.all(b[i, :] == lose):
            value = 0

    # check both diagonals
    if np.all(b.diagonal() == win) or np.all(np.fliplr(b).diagonal() == win):
        value = 1
    if np.all(b.diagonal() == lose) or np.all(np.fliplr(b).diagonal() == lose):
        value = 0

    stmap[s] = value
    return value


def disp_board(arr):
    fig, ax = plt.subplots(1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.show()


def has_winner(b):
    wonby = -1
    p0 = np.array([0] * 3)
    p1 = np.array([1] * 3)

    for i in range(3):
        # did player0 win?
        if np.all(b[:, i] == p0) or np.all(b[i, :] == p0):
            wonby = 0
        # did player1 win?
        if np.all(b[:, i] == p1) or np.all(b[i, :] == p1):
            wonby = 1

    # check both diagonals
    if np.all(b.diagonal() == p0) or np.all(np.fliplr(b).diagonal() == p0):
        wonby = 0
    if np.all(b.diagonal() == p1) or np.all(np.fliplr(b).diagonal() == p1):
        wonby = 1
    return wonby


def rand_pos():
    pos = random.randint(3, size=(2,))
    return tuple(pos)


def make_move(brd, plyr, stmap, eps=10):
    greedy = True
    txts = []
    curr_s = ":".join([str(a) for a in brd.flatten().tolist()])
    curr_v = get_state_value(curr_s, plyr, stmap)
    poss_moves = get_free_tiles(brd, plyr)
    if len(poss_moves) == 0:
        return None, None, None
    # 10% moves would be exploratory moves
    if random.random() < eps / 100.0:
        greedy = False
        next_v, next_mv, next_s = exploratory_move(brd, poss_moves, plyr, stmap)
    else:
        next_v, next_mv, next_s = greedy_move(brd, poss_moves, plyr, stmap)
    brd[next_mv] = plyr
    return curr_s, next_s, greedy


def board_disp(brd, w_count, stmap, axs, p):
    txts = []
    # -- plotting starts --
    # plot board
    im = axs["A"].matshow(brd, cmap=plt.get_cmap("Blues", 3))
    for (i, j), z in np.ndenumerate(brd):
        if z != -1:
            txts.append(axs["A"].text(j, i, z, color="green", ha="center", va="center"))
    # plot win ratio
    axs["B"].bar(
        range(len(w_count)),
        [w_count[i] for i in [0, 1, -1]],
        color="#1f77b4",
    )
    axs["B"].set_xticks([0, 1, 2], labels=["Player-0", "Player-1", "Draw"])
    # plot state
    ys = [v for k, v in sorted(stmap[0].items())]
    p.pause(0.05)
    for t in txts:
        t.remove()
    return


def play(rounds, axs, plt):
    count = 0
    while count < rounds:
        player = random.randint(2)
        player = 1
        board = init_board()
        # Play a single game to the end
        fig.suptitle(f"Round {count+1}")
        # Run a single episode
        while True:
            # one player plays i.e. either marks 1 or 0 based on the best
            # possible move from his side.
            stmp = state_map[player]
            st, st_1, greedy = make_move(board, player, stmp)
            if st_1 == None:
                # print (f"Game Over: No Winners")
                win_count[-1] += 1
                break
            # if it was a greedy move, update the state values
            if greedy:
                # update the statemap for both the players (0 & 1)
                pre_player = abs(player - 1)
                v_st = get_state_value(st, pre_player, state_map[pre_player])
                v_st_1 = get_state_value(st_1, pre_player, state_map[pre_player])
                state_map[pre_player][st] = v_st + alpha * (v_st_1 - v_st)
                v_st = get_state_value(st, player, state_map[player])
                v_st_1 = get_state_value(
                    st_1,
                    player,
                    state_map[player],
                )
                state_map[player][st] = v_st + alpha * (v_st_1 - v_st)

            winner = has_winner(board)
            if winner != -1:
                print(f"Game won by {player}!!!!")
                win_count[player] += 1
                break
            board_disp(board, win_count, state_map, axs, plt)
            player = abs(player - 1)
        print(f"Round = {count}")
        count += 1
        board_disp(board, win_count, state_map, axs, plt)
        time.sleep(1)
    pickle.dump(state_map, open("states.pl", "wb"))
    pickle.dump(win_count, open("winners.pl", "wb"))


if __name__ == "__main__":
    fig, axs = plt.subplot_mosaic(
        [["A", "B", "C"], ["A", "B", "C"]],
        figsize=(10, 3.5),
        layout="constrained",
    )
    alpha = 0.1
    _ = axs["A"].axis(False)
    play(2000, axs, plt)
