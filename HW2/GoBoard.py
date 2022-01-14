import sys
import random
import timeit
import math
import argparse
from collections import Counter
from copy import deepcopy

#from read import *
#from write import writeNextInput

class Board:
    def __init__(self):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = 5
        #self.previous_board = None # Store the previous board
        self.X_move = True # X chess plays first
        self.died_pieces = [] # Intialize died pieces to be empty
        self.moves = 0 # Trace the number of moves
        self.max_moves = 24 # The max movement of a Go game
        self.komi = 2.5 # Komi rule
        #board =   # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = [[0 for x in range(self.size)] for y in range(self.size)]
        self.previous_board = deepcopy(self.board)

    def reset_board(self):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(self.size)] for y in range(self.size)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)
        self.moves = 0
        self.died_pieces = []

    def key_state(self):
        status = []
        for i in range(self.size):
            for j in range(self.size):
                status.append("{}{}:{}-".format(i,j,self.board[i][j]))
        return "".join(status)


    def update_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def equal(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def clone(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def neighbors(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = [item for item in [(i-1, j), (i+1,j), (i, j-1), (i, j+1)] if item[0]>= 0 and item[0]<self.size and item[1]>= 0 and item[1]<self.size]
        return neighbors

    def same_color_neighbors(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.neighbors(i, j)  # Detect neighbors
        color_neighbors = []
        # Iterate through neighbors
        for item in neighbors:
            # Add to allies list if having the same color
            if board[item[0]][item[1]] == board[i][j]:
                color_neighbors.append(item)
        return color_neighbors

    def find_group_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        frontier = [(i, j)]  # stack for DFS serach
        group_pieces = []  # record allies positions during the search
        while frontier:
            piece = frontier.pop()
            group_pieces.append(piece)
            same_color_neighbors = self.same_color_neighbors(piece[0], piece[1])
            for position in same_color_neighbors:
                if position not in frontier and position not in group_pieces:
                    frontier.append(position)
        return group_pieces

    def liberty_rule(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        group = self.find_group_dfs(i, j)
        for piece in group:
            neighbors = self.neighbors(piece[0], piece[1])
            for n in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[n[0]][n[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.liberty_rule(i, j):
                        died_pieces.append((i,j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_pieces(died_pieces)
        return died_pieces

    def remove_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0

    def put_piece(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        valid_place = self.is_valid_place(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update(board)
        # Remove the following line for HW2 CS561 S2020
        self.moves += 1
        return True

    def is_valid_place(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''   
        board = self.board
        verbose = False
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False
        
        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement ({},{}). There is already a chess in this position.'.format(i,j))
            return False
        
        # Copy the board for testing
        test_go = self.clone()
        test_board = test_go.board
        oponent = 1 if piece_type == 2 else 2

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.board = test_board
        if test_go.liberty_rule(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        #print("==================== BEFORE REMOVING DIED PIECES ===================")
        #test_go.print_board()
        #print("===================================================================")
        #test_go.remove_died_pieces(oponent)
        #print("==================== AFTER REMOVING DIED PIECES ===================")
        #test_go.print_board()
        #print("===================================================================")
        if not test_go.liberty_rule(i, j):
            #if verbose:
            #print('Invalid placement ({},{}). No oponent kills found in this position with no liberty previously.'.format(i,j))
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.equal(self.previous_board, test_go.board):
                if True:
                    print('Invalid placement ({},{}). A repeat move not permitted by the KO rule.'.format(i,j))
                return False
        return True
        
    def update(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''   
        self.board = new_board

    def print_board(self):
        '''
        Visualize the board.

        :return: None
        '''
        board = self.board

        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print('-', end=' ')
                elif board[i][j] == 1:
                    print('B', end=' ')
                else:
                    print('W', end=' ')
            print()
        print('-' * len(board) * 2)

    def game_over(self, piece_type, action_move=True):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''

        # Case 1: max move reached
        if self.moves >= self.max_moves:
            return True
        # Case 2: two players all pass the move.
        if self.equal(self.previous_board, self.board) and not action_move:
            return True
        return False

    def player_pieces(self, piece_type):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''

        board = self.board
        pieces = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    pieces += 1
        return pieces          

    def game_result(self):
        '''
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''        

        pieces_1 = self.player_pieces(1)
        pieces_2 = self.player_pieces(2) + self.komi
        if pieces_1 > pieces_2: return 1
        elif pieces_1 < pieces_2: return 2
        else: return 0
        
    def start_game(self, player1, player2, verbose=False):
        '''
        The game starts!

        :param player1: Player instance.
        :param player2: Player instance.
        :param verbose: whether print input hint and error information
        :return: piece type of winner of the game (0 if it's a tie).
        '''
        self.reset_board()
        # Print input hints and error message if there is a manual player
        if player1.type == 'manual' or player2.type == 'manual':
            self.verbose = True
            print('----------Input "exit" to exit the program----------')
            print('B stands for black chess, X stands for white chess.')
            self.visualize_board()
        
        verbose = self.verbose
        # Game starts!
        while True:
            piece_type = 1 if self.X_move else 2
            oponent = 2 if self.X_move else 1
            # Judge if the game should end
            if self.game_over(piece_type):       
                result = self.game_result()
                if verbose:
                    print('Game ended.')
                    if result == 0: 
                        print('The game is a tie.')
                    else: 
                        print('The winner is {}'.format('B' if result == 1 else 'W'))
                return result

            if verbose:
                player = "B" if piece_type == 1 else "W"
                print(player + " makes move...")

            # Game continues
            if piece_type == 1: action = player1.get_input(self, piece_type)
            else: action = player2.get_input(self, piece_type)

            if verbose:
                player = "B" if piece_type == 1 else "W"
                print(action)

            if action != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not self.put_piece(action[0], action[1], piece_type):
                    if verbose:
                        self.print_board() 
                    continue

                self.died_pieces = self.remove_died_pieces(oponent) # Remove the dead pieces of opponent
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.print_board() # Visualize the board again
                print()

            self.n_moves += 1
            self.X_move = not self.X_move # Players take turn

def judge(n_move, verbose=False):

    N = 5
   
    piece_type, previous_board, board = readInput(N)
    go = Board()
    go.verbose = verbose
    go.update_board(piece_type, previous_board, board)
    go.n_moves = n_move
    try:
        action, x, y = readOutput()
    except:
        print("output.txt not found or invalid format")
        sys.exit(3-piece_type)

    if action == "MOVE":
        if not go.put_piece(x, y, piece_type):
            print('Game end.')
            print('The winner is {}'.format('B' if 3 - piece_type == 1 else 'W'))
            sys.exit(3 - piece_type)

        go.died_pieces = go.remove_died_pieces(3 - piece_type)

    if verbose:
        go.print_board()
        print()

    if go.game_over(piece_type, action):       
        result = go.game_result()
        if verbose:
            print('Game end.')
            if result == 0: 
                print('The game is a tie.')
            else: 
                print('The winner is {}'.format('B' if result == 1 else 'W'))
        sys.exit(result)

    piece_type = 2 if piece_type == 1 else 1

    if action == "PASS":
        go.previous_board = go.board
    writeNextInput(piece_type, go.previous_board, go.board)

    sys.exit(0)

def readInput(n, path="input.txt"):

    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def readOutput(path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)

def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")

def writeNextInput(piece_type, previous_board, board, path="input.txt"):
    res = ""
    res += str(piece_type) + "\n"
    for item in previous_board:
        res += "".join([str(x) for x in item])
        res += "\n"
        
    for item in board:
        res += "".join([str(x) for x in item])
        res += "\n"

    with open(path, 'w') as f:
        f.write(res[:-1]);


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--move", "-m", type=int, help="number of total moves", default=0)
    parser.add_argument("--verbose", "-v", type=bool, help="print board", default=False)
    args = parser.parse_args()

    judge(args.move, args.verbose)
        
        
