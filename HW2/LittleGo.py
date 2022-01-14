import sys
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))

from GoBoard import Board
from GoRandomPlayer import GoRandomPlayer
from GoQLearner import GoQLearner
from AlphaBeta import AlphaBeta
from tqdm import tqdm
#from PerfectPlayer import PerfectPlayer
#from SmartPlayer import SmartPlayer


PLAYER_BLACK = 1
PLAYER_WHITE = 2

def play(board, player1, player2, learn):
    """ Player 1 -> Black, Black goes first
        player 2 -> White, White has an advantage of +2.5 points
    """
    player1.set_wb(PLAYER_BLACK)
    player2.set_wb(PLAYER_WHITE)
    movement1 = "MOVE"
    movement2 = "MOVE"
    #board.visualize_board()
    while (not board.game_over(PLAYER_BLACK, movement1)) and (not board.game_over(PLAYER_WHITE, movement2)):
        m1 = player1.move(board)
        board.remove_died_pieces(player2.wb)
        #board.visualize_board()
        m2 = player2.move(board)
        print("BLACK MOVES: " + str(m1))
        print("WHITE MOVES: " + str(m2))
        board.remove_died_pieces(player1.wb)
        board.print_board()
        if (m1 == "PASS" or not m1) and (m2 == "PASS" or not m2):
            break
        if m1 == "PASS":
            if movement2 == "PASS":
                break
            movement1 = "PASS"
        else:
            movement1 = "MOVE"

        if not m2:
            if movement1 == "PASS":
                break
            movement2 = "PASS"
        else:
            movement2 = "MOVE"

        #input()
    """
    print("==============================")
    print()
    board.visualize_board()
    print("==============================")
    print()
    """
    if learn == True:
        player1.learn(board)
        player2.learn(board)

    return board.game_result()



def battle(board, player1, player2, num, learn=False, show_result=True):
    p1_stats = [0, 0, 0] # draw, win, lose
    for i in tqdm(range(0, num)):
    #for i in range(0, num):
        print("========================= NEW GAME {} ========================".format(i))
        result = play(board, player1, player2, learn)
        p1_stats[result] += 1
        board.reset_board()
        print("========================= previous winner {} ========================".format("BLACK" if result == 1 else "WHITE" if result == 2  else "TIE" ))
        if result == 2: input()

    p1_stats = [round(x / num * 100.0, 1) for x in p1_stats]
    if show_result:
        print('_' * 60)
        print('{:>15}(X) | Wins:{}% Draws:{}% Losses:{}%'.format(player1.__class__.__name__, p1_stats[1], p1_stats[0], p1_stats[2]).center(50))
        print('{:>15}(O) | Wins:{}% Draws:{}% Losses:{}%'.format(player2.__class__.__name__, p1_stats[2], p1_stats[0], p1_stats[1]).center(50))
        print('_' * 60)
        print()

    return p1_stats


if __name__ == "__main__":

    # Example Usage
    # battle(Board(show_board=True, show_result=True), RandomPlayer(), RandomPlayer(), 1, learn=False, show_result=True)
    # battle(Board(), RandomPlayer(), RandomPlayer(), 100, learn=False, show_result=True)
    # battle(Board(), RandomPlayer(), SmartPlayer(), 100, learn=False, show_result=True)
    # battle(Board(), RandomPlayer(), PerfectPlayer(), 100, learn=False, show_result=True)
    # battle(Board(), SmartPlayer(), PerfectPlayer(), 100, learn=False, show_result=True)
    qlearnerW = AlphaBeta() #GoQLearner()
    qlearnerB = AlphaBeta()
    #NUM = qlearner.GAME_NUM
    NUM = 1
    # train: play NUM games against players who only make random moves
    #print('Training QLearner against RandomPlayer for {} times......'.format(NUM))
    board = Board()
    board.print_board()
    #battle(board, GoRandomPlayer(), qlearner, NUM, learn=False, show_result=True)
    #battle(board, qlearner, GoRandomPlayer(), NUM, learn=False, show_result=True)
    # test: play 1000 games against each opponent
    print('Playing QLearner against RandomPlayer for 1000 times......')
    q_rand = battle(board, qlearnerB, GoRandomPlayer(), 20)
    #input()
    #rand_q = battle(board, qlearnerB, qlearnerW, 20)
    #print(str(qlearner.q_values))
    """
    print('Playing QLearner against SmartPlayer for 1000 times......')
    q_smart = battle(board, qlearner, SmartPlayer(), 500)
    smart_q = battle(board, SmartPlayer(), qlearner, 500)
    print('Playing QLearner against PerfectPlayer for 1000 times......')
    q_perfect = battle(board, qlearner, PerfectPlayer(), 500)
    perfect_q = battle(board, PerfectPlayer(), qlearner, 500)
    """
    # summarize game results
    winning_rate_w_random_player  = round(40 -  (q_rand[2] + rand_q[1]) / 2, 2)
    #winning_rate_w_smart_player   = round(100 - (q_smart[2] + smart_q[1]) / 2, 2)
    #winning_rate_w_perfect_player = round(100 - (q_perfect[2] + perfect_q[1]) / 2, 2)

    print("Summary:")
    print("_" * 60)
    print("QLearner VS  RandomPlayer |  Win/Draw Rate = {}%".format(winning_rate_w_random_player))
    #print("QLearner VS   SmartPlayer |  Win/Draw Rate = {}%".format(winning_rate_w_smart_player))
    #print("QLearner VS PerfectPlayer |  Win/Draw Rate = {}%".format(winning_rate_w_perfect_player))
    print("_" * 60)

    grade = 0
    if winning_rate_w_random_player >= 85:
        grade += 25 if winning_rate_w_random_player >= 95 else winning_rate_w_random_player * 0.15
    """
    if winning_rate_w_smart_player >= 85:
        grade += 25 if winning_rate_w_smart_player >= 95 else winning_rate_w_smart_player * 0.15
    if winning_rate_w_perfect_player >= 85:
        grade += 20 if winning_rate_w_perfect_player >= 95 else winning_rate_w_perfect_player * 0.10
    grade = round(grade, 1)
    """
    print("\nTask 2 Grade: {} / 70 \n".format(grade))

#   output_file = sys.argv[1]
#    with open(output_file, 'w') as f:
#        f.write(str(grade) + '\n')

