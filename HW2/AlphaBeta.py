import numpy as np
from queue import PriorityQueue
import random
import time
import math
from GoBoard import Board
from copy import deepcopy
"""
Class Board is based on the GO class in the provided host.py

Some similarities may arise, but as stated by TAs n in question @277 it is OK if functionalities and logic 
seem similar.
"""
class Board:
    def __init__(self):
        """
        Board for the GO game and all the logistics.
        """
        self.size = 5 # Size of the board
        #self.previous_board = None # Store the previous board
        self.X_move = True # X chess plays first
        self.died_pieces = [] # Intialize died pieces to be empty
        self.moves = 0 # Count the total number of moves in the game
        self.max_moves = 24 # The maximum number of movements permitted
        self.komi = 2.5 # Komi rule
        #board =   # Empty space represented with 0
        # Black pieces represented with 1
        # White pieces represented with 2
        self.board = [[0 for x in range(self.size)] for y in range(self.size)]
        self.previous_board = deepcopy(self.board)

    def reset_board(self):
        '''
        Reset the board.

        :return: None.
        '''
        board = [[0 for x in range(self.size)] for y in range(self.size)]  # Put all positions of board to 0 (empty)

        self.board = board # Set the board to the empty board
        self.previous_board = deepcopy(board) # Set previous board to empty also
        self.moves = 0 # Reset the number of moves
        self.died_pieces = [] # Reset the tracking of died pieces

    def key_state(self):
        """
        Represent a state with a string for Q Learning

        :return: The string key
        """
        status = []
        for i in range(self.size):
            for j in range(self.size):
                status.append("{}{}:{}-".format(i,j,self.board[i][j]))
        return "".join(status)


    def update_board(self, piece_type, previous_board, board):
        '''
        Update the status of the board.
        :param piece_type: the id of the player (either 1 or 2 for black or white, respectively)
        :param previous_board: representation of the board in the previous movement.
        :param board: current board.
        :return: None.
        '''

        for i in range(self.size): # For every row
            for j in range(self.size): # For every column
                # If the position had a piece of the player previously and not now, we have detected a dead piece.
                if board[i][j] != piece_type and previous_board[i][j] == piece_type: 
                    self.died_pieces.append((i, j))

        self.previous_board = previous_board
        self.board = board

   
    def equal(self, board1, board2):
        """
        Compare two boards.

        :param board1: A board.
        :param board2: Another board.
        :return: True if both boards are the same, False otherwise.
        """
        for i in range(self.size): # For every row
            for j in range(self.size): # For every column
                if board1[i][j] != board2[i][j]: # If the content of both boards is not the same at every position
                    return False # Boards differ
        return True # Boards are equal

    """
    def clone(self):
        '''
        Clone the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)
    """

    def neighbors(self, i, j):
        '''
        Looks for the neigbors of position (i,j).

        :param i: row coordinate.
        :param j: column coordinate.
        :return: list with the neigbors of (i,j).
        '''
        neighbors = [item for item in [(i-1, j), (i+1,j), (i, j-1), (i, j+1)] if item[0]>= 0 and item[0]<self.size and item[1]>= 0 and item[1]<self.size]
        return neighbors

    def same_color_neighbors(self, i, j):
        '''
        Find the neighbors pertaining to the same player.

        :param i: row coordinate.
        :param j: column coordinare.
        :return: neighbors of the same color as (i,j).
        '''
        board = self.board
        neighbors = self.neighbors(i, j)  # Detect neighbors of the position
        color_neighbors = []
        # For each neighbor
        for item in neighbors:
            # Check if neighbor and (i,j) pertain to the same player
            if board[item[0]][item[1]] == board[i][j]:
                color_neighbors.append(item)
        return color_neighbors

    def find_group_dfs(self, i, j):
        '''
        Use DFS to find the group of pieces to which (i,j) pertains.

        :param i: row coordinate.
        :param j: column coordinate.
        :return: List with all the pieces in the group.
        '''
        frontier = [(i, j)]  # The DFS frontier
        group_pieces = []  # The pieces in the group
        while frontier: # For each position
            piece = frontier.pop()
            group_pieces.append(piece) # Add position to the group 
            same_color_neighbors = self.same_color_neighbors(piece[0], piece[1]) # Obtain the neighbors pertaining to the same player
            for position in same_color_neighbors: # For each neighbor of the same player
                if position not in frontier and position not in group_pieces: # If position not in group and not already queued in the frontier
                    frontier.append(position) # Add to the frontier to explore it
        return group_pieces

    def liberty_rule(self, i, j):
        '''
        Apply the liberty rule to position (i, j).

        :param i: row coordinate.
        :param j: column coordinate.
        :return: True if the position has liberty, False otherwise.
        ''' 
        group = self.find_group_dfs(i, j)
        for piece in group:
            neighbors = self.neighbors(piece[0], piece[1])
            for n in neighbors: # For every neighbor
                # If there is empty space around a piece, it has liberty
                if self.board[n[0]][n[1]] == 0:
                    return True
        # Otherwise, it has no liberty
        return False

    def find_died_pieces(self, player):
        '''
        Detect all the died pieces of a player.

        :param player: Black (1) or White (2).
        :return: List with the positions of the died pieces.
        '''
        died_pieces = []

        for row in range(self.size):
            for col in range(self.size):
                # Check if there is a piece at this position:
                if self.board[row][col] == player:
                    # The piece die if it has no liberty
                    if not self.liberty_rule(row, col):
                        died_pieces.append((row,col))
        return died_pieces
    
    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces_player = self.find_died_pieces(piece_type)
        if not died_pieces_player: return []
        self.remove_pieces(died_pieces_player)
        return died_pieces_player
    
    def remove_pieces(self, positions):
        '''
        Remove pieces. If flag 'died' is True, then remove died pieces of player.

        :param positions: positions of the pieces to remove
        :param died: True if we are removing died pieces. False otherwise
        :param player: If died is True, player from who we want to remove the died pieces.
        :return: None.
        '''
        
        for piece in positions:
            self.board[piece[0]][piece[1]] = 0

    def put_piece(self, row, col, player):
        '''
        Put a piece in the board.

        :param row: row coordinate.
        :param col: column coordinate.
        :param player: Black (1) or White (2).
        :return: True if the it is a valid move, false otherwise.
        '''

        valid_place = self.is_valid_place(row, col, player)
        if not valid_place:
            return False
        self.previous_board = deepcopy(self.board)
        self.board[row][col] = player
        self.update_board(player, self.previous_board, self.board)
        # Remove the following line for HW2 CS561 S2020
        self.moves += 1
        return True

    def is_valid_place(self, row, col, player):
        '''
        Check whether a placement is valid.

        :param row: row coordinate.
        :param col: column coordinate.
        :param player: Black (1) or White (2).
        :return: True if the position (row, col) is a valid position for player to put a piece.
        '''   
        verbose = False
        
        # Check if the place is in the board range
        if not (row >= 0 and row < len(board)):
            """
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            """
            return False
        if not (col >= 0 and col < len(board)):
            """"
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            """
            return False
        
        # Check if the place already has a piece
        if self.board[row][col] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False
        
        # Copy the board for testing
        test_go = deepcopy(self)
        test_board = test_go.board
        oponent = 1 if player == 2 else 2

        # Check if the place has liberty
        test_board[row][col] = player
        test_go.board = test_board
        if test_go.liberty_rule(row, col):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(oponent)
        if not test_go.liberty_rule(row, col):
            #if verbose:
            #print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.equal(self.previous_board, test_go.board):
                if True:
                    print('Invalid placement {}. A repeat move not permitted by the KO rule.'.format((i,j)))
                return False
        return True
        

    def print_board(self):
        '''
        Print the board.

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

    def game_over(self, player, action_move=True):
        '''
        Check if the game is over.

        :param player: Black (1) or White (2).
        :param action_move: True if there is an actual move. False if it there is a "PASS" move.
        :return: True if the game is over, False otherwise.
        '''

        # Game over condition 1: We have passed the maximum of 24 moves
        if self.moves >= 24:
            return True
        # Game over condition 1: We have played a PASS move twice
        if self.equal(self.previous_board, self.board) and not action_move:
            return True

        return False

    def player_pieces(self, player):
        '''
        Get score of a player by counting the number of stones.

        :param player: Black (1) or White (2)
        :return: The number of pieces of player in the board.
        '''

        pieces = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == player:
                    pieces += 1
        return pieces          

    def game_result(self):
        '''
        The result of the game when it's over.

        :return: 1 if Black wins, 2 if White wins, 0 otherwise
        '''        

        pieces_1 = self.player_pieces(1) # Black points equals the number of Black pieces
        pieces_2 = self.player_pieces(2) + 2.5 # White points equals the number of White pieces plus the komi of 2.5
        if pieces_1 > pieces_2: return 1 # Black wins
        elif pieces_1 < pieces_2: return 2 # White wins
        else: return 0 # A tie
        
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


class AlphaBeta:

    def __init__(self):
        self.type = 'random'
        self.wb = None

    def set_wb(self, wb):
        self.wb = wb

    def move(self, game):
        move = make_move(game, self.wb)
        if move == "PASS" or move is None:
            if self.wb == 1:
                print("BLACK MOVES: PASS")
            else:
                print("WHITE MOVES: PASS")
            return False
            game.moves += 1
        else:
            game.put_piece(move[0], move[1], self.wb)
            if self.wb == 1:
                print("BLACK MOVES: ({},{})".format(move[0], move[1]))
            else:
                print("WHITE MOVES: ({},{})".format(move[0], move[1]))
            return True

def check_available_positions(game, player=0):
    """
    Check available positions, it is, empty squares. This function can be used also for counting the number of pieces of a player.

    :param game: The game object
    :param player: Default 0 (counts the empty squares), 1 for counting Black pieces and 2 for counting White pieces
    :return: List with the positions
    """
    available_positions = []
    for i in range(game.size):
        for j in range(game.size):
            if game.board[i][j] == player:
                available_positions.append((i,j))
    return available_positions

def check_valid_positions(game, player):
    """
    Check which empty squares are valid positions for a given player.

    :param game: The game object
    :param player: Black (1) or White (2)

    :return: A list with the valid positions for player
    """
    available_positions = check_available_positions(game)
    valid_positions = []
    for position in available_positions:
        if game.is_valid_place(position[0], position[1], player):
            valid_positions.append(position)
    random.shuffle(valid_positions)
    return valid_positions

def strong_group(game, player):
    """
    Find groups of pieces of a given player and sort them in increasing order of group size

    :param game: The game object
    :param player: Black (1) or White (2)

    :return: The PriorityQueue of the groups 
    """
    available_positions = check_valid_positions(game, player)
    board = game.board
    oponent = 1 if player == 2 else 2

    groups = PriorityQueue()
    for row in range(game.size):
        for col in range(game.size):
            if game.board[row][col] == player:
                pieces_group = game.find_group_dfs(row, col)
                if len(pieces_group) > 0:
                    groups.put((len(pieces_group), pieces_group))
    return groups

def strong_group_inverted(game, player):
    """
    Find groups of pieces of a given player and sort them in decreasing order of group size

    :param game: The game object
    :param player: Black (1) or White (2)

    :return: The PriorityQueue of the groups 
    """
    available_positions = check_valid_positions(game, player)
    board = game.board
    oponent = 1 if player == 2 else 2

    groups = PriorityQueue()
    for row in range(game.size):
        for col in range(game.size):
            if game.board[row][col] == player:
                pieces_group = game.find_group_dfs(row, col)
                if len(pieces_group) > 0:
                    groups.put((-len(pieces_group), pieces_group))
    return groups

def find_group(game, row, col, player):
    """
    Find the neighbors of (row, col) if there is a piece of player in that position

    :param game: The game object
    :param row: Row coordinate
    :param col: Column coordinate
    :param player: Black (1) or White (2)

    :return: The list with the neigbors of (row, col), None otherwise
    """
    if game.board[row][col] == player:
        return game.neighbors(row, col)

def complete_diagonals(game, player):
    """
    Find diagonals that player partially owns and try to complete the most completed ones

    :param game: The game object
    :param player: Black (1) or White (2)
    """
    diagonals_indices = [[(3,0), (4,1)], [(2,0), (3,1), (4,2)], [(1,0), (2,1), (3,2), (4,3)], [(0,0), (1,1), (2,2), (3,3), (4,4)],
                         [(0,1), (1,2), (2,3), (3,4)], [(0,2), (1,3), (2, 4)], [(0,3), (1,4)], [(0,1), (1, 0)], 
                         [(0,2), (1,1), (2,0)], [(0,3), (1,2), (2,1), (3,0)], [(0,4), (1,3), (2,2), (3,1), (4,0)],
                         [(1,4), (2,3), (3,2), (4,1)], [(2,4), (3,3), (4,2)], [(3,4), (4,3)]]

    best_moves = PriorityQueue()
    for diagonal in diagonals_indices:
        items = len(diagonal)
        positions = []
        flag = True
        for element in diagonal:
            if game.board[element[0]][element[1]] != player:
                if game.board[element[0]][element[1]] == 0:
                    items -= 1
                    positions.append(element)
                else:
                    flag = False
                    break
        if flag:
            if len(diagonal)-items == 1:
                best_moves.put((len(diagonal)-items, positions))

    return best_moves



def conquered_areas(game, wb):
    
    free_groups = []
    visited_points = []
    conquered_groups = []
    for i in range(game.size):
        for j in range(game.size):
            if (i,j) not in visited_points:
                group = find_group(game, i, j, 0)
                if group is not None:
                    free_groups.append(group)
                    for position in group:
                        visited_points.append(position)
    for group in free_groups:
        flag = True
        for position in group:
            neighbors = game.neighbors(position[0], position[1])
            for neighbor in neighbors:
                if game.board[neighbor[0]][neighbor[1]] != wb:
                    flag = False
                    break
            if not flag:
                break
        conquered_groups += group

    return group


def increase_strong_group(game, player):
    """
    Find the least strong group and increase it to make it stronger.

    :param game: The game object
    :param player: Black (1) or White (2)

    :return: Best position to strenghten a group or None if we better decide there it is not worth to increase the size of any group
    """

    groups = strong_group(game, player)
    conquered_positions = conquered_areas(game, player)
    print("GREAT GROUPS: \n{}".format(str(groups.queue)))
    if len(groups.queue) == 0:
        return None
    pieces = groups.get()
    if pieces[0] > 6:
        return None
    pieces = pieces[1]
    possible_positions = []
    for position in pieces:
        neighbors = game.neighbors(position[0], position[1])
        for neighbor in neighbors:
            if game.board[neighbor[0]][neighbor[1]] == 0:
                possible_positions.append(neighbor)
    
    if len(possible_positions) > 0:
        for position in possible_positions:
            if position[0] == 0 or position[0] == 4 or position[1] == 0 or position[1] == 4 and position not in conquered_positions:
                return position
        random.shuffle(possible_positions)
        return possible_positions[0]
    else:
        return None

def attack_action(game, player):
    """
    Find the best attacking positions. An attack position is a position in which we can place
    one of our pieces to kill an oponent's piece or group.

    :param game: The game object
    :param player: Black (1) or White (2)

    :return: A list with the best attacking positions
    """

    #available_positions = []
    available_positions = check_available_positions(game)
    board = game.board
    oponent = 1 if player == 2 else 2
    """
    for i in range(game.size):
        for j in range(game.size):
            if board[i][j] == 0:
                available_positions.append((i,j))
    """
    #print("Possible moves for attacking:\n" + str(available_positions))
    best_moves = PriorityQueue()
    for position in available_positions:
        board[position[0]][position[1]] = player
        oponent_died_pieces = game.find_died_pieces(oponent)
        board[position[0]][position[1]] = 0
        if len(oponent_died_pieces) >= 1:
            best_moves.put((-len(oponent_died_pieces), position))
    return_moves = []
    while len(best_moves.queue) > 0:
        move = best_moves.get()[1]
        board[move[0]][move[1]] = player
        equal_boards = game.equal(game.previous_board, board)
        board[move[0]][move[1]] = 0
        if not equal_boards:
            return_moves.append(move)
    return return_moves


def block_kills_action(game, player):

    """
    We find positions were our piece can be killed next ply, but that kill some oponent's pieces in this ply.

    :param game: The game object
    :param player: Black (1) or White (2)

    :return: The positions in a list if any, or an empty list otherwise.
    """
    new_game = deepcopy(game)
    available_positions = check_valid_positions(new_game, player)
    oponent = 1 if player == 2 else 2

    blocked_movements = []
    for position in available_positions:
        new_game.board[position[0]][position[1]] = player
        oponent_kills = new_game.find_died_pieces(oponent)
        new_game.remove_died_pieces(oponent)
        oponent_available_positions = check_valid_positions(new_game, oponent)
        for op_position in oponent_available_positions:
            new_game.board[op_position[0]][op_position[1]] = oponent
            our_died_pieces = new_game.find_died_pieces(player)
            new_game.board[op_position[0]][op_position[1]] = 0
            if position in our_died_pieces:# and len(oponent_kills) < 1:
                blocked_movements.append(position)
        new_game.board[position[0]][position[1]] = 0

    return blocked_movements

def defense_action(game, player):
    """
    Positions to put a player piece in to avoid a kill next ply.
    
    :param game: The game object
    :param player: Black (1) or White (2)

    :return: The list with such movements sorted by potential number of killed pieces.
    """
    board = game.board
    available_positions = check_available_positions(game)
    oponent = 1 if player == 2 else 2

    movements = PriorityQueue()
    for position in available_positions:
        board[position[0]][position[1]] = oponent
        our_died_pieces = game.find_died_pieces(player)
        if len(our_died_pieces) >= 1:
            movements.put((-len(our_died_pieces), position))
        board[position[0]][position[1]] = 0

    return movements

def less_liberty_action(game, player, good_moves):

    """
    Positions that substract liberty from oponent.

    :param game: The game object
    :param player: Black (1) or White (2)
    :param good_moves: A list with the valid moves for player

    :return: A movement to restrict oponent's liberty and that is valid for player if it exists. None, otherwise.
    """    
    
    board = game.board
    oponent = 1 if player == 2 else 2
    oponent_positions = check_available_positions(game, oponent)

    liberty_oponent_positions = []
    for position in oponent_positions:
        for add in [(1,0), (-1,0), (0,1), (0,-1)]:
            x = position[0] + add[0]
            y = position[1] + add[1]
            if x >= 0 and x < game.size and y >= 0 and y < game.size and board[x][y] == 0:
                liberty_oponent_positions.append((x,y))


    for move in good_moves:

        in_vitro_game = deepcopy(game)
        in_vitro_board = in_vitro_game.board 
        in_vitro_board[move[0]][move[1]] = player

        oponent_deaths = in_vitro_game.find_died_pieces(oponent)
        for position in oponent_deaths:
            in_vitro_board[position[0]][position[1]] = 0

        new_oponent_positions = check_available_positions(in_vitro_game, oponent)
        new_liberty_oponent_positions = []
        for position in new_oponent_positions:
            for add in [(1,0), (-1,0), (0,1), (0,-1)]:
                x = position[0] + add[0]
                y = position[1] + add[1]
                if x >= 0 and x < in_vitro_game.size and y >= 0 and y < in_vitro_game.size and in_vitro_board[x][y] == 0:
                    new_liberty_oponent_positions.append((x,y))

        if len(liberty_oponent_positions) - len(new_liberty_oponent_positions) > 0:
            return move

    return None

def strategic_opening_positions(game, good_moves):
    """
    Best opening positions according to some game masters.

    :param game: The game object
    :param good_moves: List of valid moves

    :return: A strategic and valid position if it exists, None otherwise
    """
    priority_strategic_positions = [(2,2), (1,1), (1,3), (3,1), (3,3), (0,2), (4,2), (2,0), (2,4)]

    for position in priority_strategic_positions:
        if position in good_moves:
            return position

    return None



def minimax(game, player, good_moves):

    """
    Apply minimax with limited depth for detecting the best movement of the oponent and try to save it. If we cannot save it
    by hand, then apply minimax with limited time for detecting our best movement at that time.

    :param game: The game object
    :param player: Black (1) or White (2)
    :param good_moves: Valid movements for player

    :return: Best movement found given the depth and time limits.
    """

    oponent = 1 if player == 2 else 2 # Set player's oponent
    oponent_game = deepcopy(game) # Clone the game object for simulating games without changing the original game
    oponent_game.update_board(oponent, oponent_game.previous_board, oponent_game.board)

    move, points = minimax_tree(oponent_game, oponent, True) # Launch a minimax tree for the oponent to see what is their best move

    new_game = deepcopy(game)
    new_game.board[move[0]][move[1]] = oponent # Perform the best move for the oponent

    available_positions = check_available_positions(new_game) # See the valid positions for player in the board after the movement

    priority_min_node = {}

    for position in available_positions: # For each valid position
        new_game.board[position[0]][position[1]] = player # Place the piece at the valid positions
        oponent_died_pieces = new_game.find_died_pieces(oponent) # Count the killed oponent's pieces
        new_game.board[position[0]][position[1]] = 0 # Leave the board as it was before the player placecment
        if len(oponent_died_pieces) > 0: # If we kill any oponent's piece
            priority_min_node[position] = len(oponent_died_pieces) # Add it to the priority queue

    sorted_priority_min_node = sorted(priority_min_node, key = priority_min_node.get, reverse = True)
    new_game.board[move[0]][move[1]] = 0 # Restore the original board
    oponent_moves_to_remove = []

    for position in sorted_priority_min_node: # For each killing oponent position

        new_game.board[position[0]][position[1]] = player # Place the player in that position
 
        #oponent_available_positions = check_valid_positions(new_game, oponent) # Check the oponent valid positions
        oponent_available_positions = check_available_positions(new_game) # Check the oponent valid positions

        for x, y in oponent_available_positions: # For each valid oponen position
            new_game.board[x][y] = oponent # Place oponent piece in that position
            oponent_died_pieces = new_game.find_died_pieces(player) # Count player's killed pieces
            
            if position in oponent_died_pieces: #and positions_min_node[position] - len(oponent_died_pieces) < 1: # If the piece of player is killed
                oponent_moves_to_remove.append(position) # Try to avoid playing that move

            new_game.board[x][y] = 0 # Restore original board

    for position in oponent_moves_to_remove: # For each position to avoid
        if position in sorted_priority_min_node: # If the position killed some oponent
            sorted_priority_min_node.remove(position) # Remove position from possible positions


    for position in sorted_priority_min_node: # For each valid position
        if position in good_moves: # If it is in good_moves
            return position # Play it

    # Otherwise, run minimax as deep as possible to select the best movement
    position, points = minimax_tree(game, player)
    return position

def max_node(game, player, max_depth, alpha, beta, init_time):
    """
    Create and expand a max node in the minimax tree.

    :param game: The game object
    :param player: Black (1) or White (2)
    :param max_depth: The maximum depth permitted from the current node
    :param alpha: Alpha value for a/b prunning
    :param beta: Beta value for a/b prunning
    :param init_time: Time when the minimax tree started expanding

    :return: Best movement from the node's children or None if it's a leaf node. Also the best value of the children, or the 
    points earned by the player if it is a leaf node.
    """
    #print("MAX DEPTH: {}".format(max_depth))
    oponent = 1 if player == 2 else 2 # Set the player's oponent
    in_vitro_game = deepcopy(game) # Clone the game to not mess it up
    value = -math.inf # Set the initial value of the node. Max node, so -inf
    #available_positions = check_valid_positions(in_vitro_game, player) # Check valid positions for player
    available_positions = check_available_positions(in_vitro_game) # Check valid positions for player

    avoid_positions = []

    for position in available_positions: # For each valid positions
        in_vitro_game.board[position[0]][position[1]] = player # Play in the position
        #oponent_positions = check_valid_positions(in_vitro_game, oponent) # See valid positions for the oponent
        oponent_positions = check_available_positions(in_vitro_game) # See valid positions for the oponent

        for oponent_position in oponent_positions: # For each oponent valid position
            in_vitro_game.board[oponent_position[0]][oponent_position[1]] = oponent # Place one of their pieces
            our_deaths = in_vitro_game.find_died_pieces(player) # Count player's died pieces
            in_vitro_game.board[oponent_position[0]][oponent_position[1]] = 0 # Restore game
            if position in our_deaths: # If the player selected position is killed
                if position not in avoid_positions:
                    avoid_positions.append(position) # Avoid position

        in_vitro_game.board[position[0]][position[1]] = 0 # Restore game

    for position in avoid_positions: # For each position to avoid
        if position in available_positions: # If it is also valid
            available_positions.remove(position) # Remove it

    killers = attack_action(game, player)
    killers.reverse()

    liberty = less_liberty_action(game, player, available_positions)
    if liberty in available_positions:
        available_positions.remove(liberty)
        available_positions = [liberty] + available_positions

    for position in killers:
        if position in available_positions:
            available_positions.remove(position)
            available_positions = [position] + available_positions

    finish_time = time.time() # Current time

    if len(available_positions) == 0 or max_depth == 0 or finish_time - init_time > 8.5: # If no possible action, or depth/time limits reached
        if finish_time-init_time > 8.5:
            print("TIME LIMIT")
        return (-1,-1), in_vitro_game.player_pieces(player) - in_vitro_game.player_pieces(oponent)
        """
        if player == 1:
            return (-1,-1), in_vitro_game.player_pieces(player)
        else:
            return (-1,-1), in_vitro_game.player_pieces(player)+2.5
        """
    else: # Otherwise, we expand the node
        for position in available_positions: # For each valid position
            new_game = deepcopy(game) # Clone the original game
            new_game.put_piece(position[0], position[1], player) # Place the player's pice in the position
            new_game.remove_died_pieces(oponent) # Remove the dead oponent's pieces
            # Set following player
            if player == 1:
                next_player = 2
            else:
                next_player = 1

            min_move, min_score = min_node(new_game, next_player, max_depth-1, alpha, beta, init_time) # Expand min children nodes
            #Alpha/Beta pruning
            if min_score > value: # if the score of the min children is greater than the current value   
                value = min_score # Update the value
                best_move = position # update the position related to the new best value
                
            elif min_score == value and position in [(2,2), (1,1), (1,3), (3,1), (3,3), (0,2), (4,2), (2,0), (2,4)]:
                print("Same score but with preferred position: {}".format(str(position)))
                value = min_score # Update the value
                best_move = position # update the position related to the new best value
            
            alpha = max(min_score, alpha) # Set alpha to the greatest of the min_score or alpha
            if beta <= alpha: # If beta is lower or equal than the actual alpha, we prune
                print("Prune max node at level {}".format(max_depth))
                break
        return best_move, value

def min_node(game, player, max_depth, alpha, beta, init_time):
    """
    Create and expand a min node in the minimax tree.

    :param game: The game object
    :param player: Black (1) or White (2)
    :param max_depth: The maximum depth permitted from the current node
    :param alpha: Alpha value for a/b prunning
    :param beta: Beta value for a/b prunning
    :param init_time: Time when the minimax tree started expanding

    :return: Best movement from the node's children or None if it's a leaf node. Also the best value of the children, or the 
    points earned by the player if it is a leaf node.
    """
    #print("MIN DEPTH: {}".format(max_depth))
    value = math.inf # Set value. Min value so initial value is +inf.
    oponent = 1 if player == 2 else 2 # Set player's oponent
    new_game = deepcopy(game) # Clone game so we don't mess up
    #min_moves = check_valid_positions(game, player) # Check the valid positions for the player
    min_moves = check_available_positions(game) # Check the valid positions for the player

    killers = attack_action(game, player)
    killers.reverse()

    liberty = less_liberty_action(game, player, min_moves)
    if liberty in min_moves:
        min_moves.remove(liberty)
        min_moves = [liberty] + min_moves

    for position in killers:
        if position in min_moves:
            min_moves.remove(position)
            min_moves = [position] + min_moves

    finish_time = time.time() # Current time

    if len(min_moves) == 0 or max_depth == 0 or finish_time-init_time > 8.5: # If no moves available or depth/time limits reached
        if finish_time-init_time > 8.5:
            print("TIME LIMIT")
        return (-1,-1), new_game.player_pieces(oponent) - new_game.player_pieces(player)
        """
        if player == 1:
            return (-1,-1), new_game.player_pieces(player)
        else:
            return (-1,-1), new_game.player_pieces(player)+2.5
        """
    else:
        for position in min_moves: # For each valid position
            min_game = deepcopy(game) # Clone the game
            min_game.put_piece(position[0], position[1], player) # Place the player's piece
            min_game.remove_died_pieces(oponent) # Remove oponent's died pieces
            # Set next player
            if player == 1:
                next_player = 2
            else:
                next_player = 1
            # Expand the node
            max_move, max_score = max_node(new_game, next_player, max_depth-1, alpha, beta, init_time) 

            if max_score < value: # If the score obtained is lower than the current value
                value = max_score # Update value
                best_move = position # update the position related to the new value
            beta = min(max_score, beta) # Set beta to the lower of max_score or current beta
         
            if beta <= alpha: # If beta is lowe or equal than alpha, prune
                print("Prune min node at level {}".format(max_depth))
                break
        return best_move, value

def minimax_tree(game, player, oponent=False):
    """
    Expand the minimax tree.

    :param game: The game object
    :param player: Black (1) or White (2)
    :param oponent: True if we expand a tree for looking at the best movement of our oponent (aggressive player).
    False for player's normal tree.
    :return: best move for player.
    """
    init_time = time.time() # Current time at the begining of the tree expansion
    if oponent:
        move, points = max_node(game, player, 2, -math.inf, math.inf, init_time)

    else:
        move, points = max_node(game, player, min(5, 24-game.moves), -math.inf, math.inf, init_time)
    return move, points

def make_move_black(game, player):
    game.print_board()
    print(type(game))
    # Locate the player's valid positions
    my_possible_moves = check_available_positions(game)
    print("POSSIBLE MOVES: {}".format(str(my_possible_moves)))

    # Locate the player's positions to avoid due to ensured kill in the next ply
    avoid_moves = block_kills_action(game, player)
    print("Moves to avoid: \n" + str(avoid_moves))
    for move in avoid_moves: # Remove shuch those movements from the possible moves
        if move in my_possible_moves:
            my_possible_moves.remove(move)

    # PASS if there are no movements possible.
    if len(my_possible_moves) == 0:
        return "PASS"
    
    ################### GAME OPENING ###################
    if len(my_possible_moves) >= 20:
        
        print("Deciding attack actions...")
        # Locate attacking valid positions
        actions = attack_action(game, player)
        print("Attack actions: \n{}".format(str(actions)))
        for action in actions:
            if action in my_possible_moves:
                return action # Attack if possible
        
        action = strategic_opening_positions(game, my_possible_moves)
        print("Strategic opening movement: {}".format(action))
        if action is not None and action in my_possible_moves:
            return action # Play such action if found

    ################### GAME OPENING ###################
    
    print("Deciding attack actions...")
    # Locate attacking valid positions
    actions = attack_action(game, player)
    print("Attack actions: \n{}".format(str(actions)))
    for action in actions:
        if action in my_possible_moves:
            return action # Attack if possible
    """
    ################### GAME MIDDLE ####################
    if len(my_possible_moves) > 10:
        print("Deciding attack actions...")
        # Locate attacking valid positions
        actions = attack_action(game, player)
        print("Attack actions: \n{}".format(str(actions)))
        for action in actions:
            if action in my_possible_moves:
                return action # Attack if possible

        #Locate defense valid positions
        defense_moves = defense_action(game, player)
        print("Defense moves: \n{}".format(defense_moves.queue))

        while len(defense_moves.queue) > 0:
            action = defense_moves.get()[1]
            if action in my_possible_moves:
                return action # Defense the biggest number of our pieces if possible

        # Locate positions to increase our conquered space with the least number of pieces possible
        diagonals = complete_diagonals(game, player)
        print("DIAGONALS TO FILL: \n{}".format(str(diagonals.queue)))
        while len(diagonals.queue) > 0:
            action = diagonals.get()[1]
            if len(action) == 1:
                action = action[0]
                print("Selected diagonal position: {}".format(str(action)))
                if action in my_possible_moves:
                    return action # Increase the chances of conquering if possible

        # Locate oponent liberty restraining position
        action = less_liberty_action(game, player, my_possible_moves)
        print("Best less liberty action: {}".format(action))
        if action is not None:
            return action # Restrict oponent's liberty if possible

        # Locate positions to increase our conquered space with the least number of pieces possible
        diagonals = complete_diagonals(game, player)
        print("DIAGONALS TO FILL: \n{}".format(str(diagonals.queue)))
        while len(diagonals.queue) > 0:
            action = diagonals.get()[1]
            if len(action) == 1:
                action = action[0]
                print("Selected diagonal position: {}".format(str(action)))
                if action in my_possible_moves:
                    return action # Increase the chances of conquering if possible

        # Locate positions where we increase the strenght of our weakest group
        action = increase_strong_group(game, player)
        print("Move to increase a group: \n{}".format(str(action)))
        if action is not None and action in my_possible_moves:
            return action # Increase such strenght if possible
    
    ################### GAME MIDDLE ####################
    """
    ################ GAME CLOSE TO END #################
    print("============ MINIMAX !!!!!!!")
    action = minimax(game, player, my_possible_moves) # Very little positions available, so rin minimax in a controlled breadth scenario
    if action == (-1,-1):
        return None
    return action

    if action in my_possible_moves:
        return action
    return None
    ################ GAME CLOSE TO END #################


def make_move_white(game, player):
    game.print_board()
    print(type(game))
    # Locate the player's valid positions
    my_possible_moves = check_valid_positions(game, player)
    print("POSSIBLE MOVES: {}".format(str(my_possible_moves)))

    # Locate the player's positions to avoid due to ensured kill in the next ply
    avoid_moves = block_kills_action(game, player)
    print("Moves to avoid: \n" + str(avoid_moves))
    for move in avoid_moves: # Remove shuch those movements from the possible moves
        if move in my_possible_moves:
            my_possible_moves.remove(move)

    # PASS if there are no movements possible.
    if len(my_possible_moves) == 0:
        return "PASS"

    ################### GAME OPENING ###################
    if len(my_possible_moves) >= 20:
        print("Deciding attack actions...")
        # Locate attacking valid positions
        actions = attack_action(game, player)
        print("Attack actions: \n{}".format(str(actions)))
        for action in actions:
            if action in my_possible_moves:
                return action # Attack if possible

        # Look for best strategic opening position
        action = strategic_opening_positions(game, my_possible_moves)
        print("Strategic opening movement: {}".format(action))
        if action is not None and action in my_possible_moves:
            return action # Play such action if found
        
    ################### GAME OPENING ###################
    
    ################### GAME MIDDLE ####################
    print("Deciding attack actions...")
    # Locate attacking valid positions
    actions = attack_action(game, player)
    print("Attack actions: \n{}".format(str(actions)))
    for action in actions:
        if action in my_possible_moves:
            return action # Attack if possible
    

    #Locate defense valid positions
    defense_moves = defense_action(game, player)
    print("Defense moves: \n{}".format(defense_moves.queue))

    while len(defense_moves.queue) > 0:
        action = defense_moves.get()[1]
        if action in my_possible_moves:
            return action # Defense the biggest number of our pieces if possible

    # Locate oponent liberty restraining position
    action = less_liberty_action(game, player, my_possible_moves)
    print("Best less liberty action: {}".format(action))
    if action is not None:
        return action # Restrict oponent's liberty if possible
    
    # Locate positions to increase our conquered space with the least number of pieces possible
    diagonals = complete_diagonals(game, player)
    print("DIAGONALS TO FILL: \n{}".format(str(diagonals.queue)))
    while len(diagonals.queue) > 0:
        action = diagonals.get()[1]
        if len(action) == 1:
            action = action[0]
            print("Selected diagonal position: {}".format(str(action)))
            if action in my_possible_moves:
                return action # Increase the chances of conquering if possible

    # Locate positions where we increase the strenght of our weakest group
    action = increase_strong_group(game, player)
    print("Move to increase a group: \n{}".format(str(action)))
    if action is not None and action in my_possible_moves:
        return action # Increase such strenght if possible
    ################### GAME MIDDLE ####################
    
    ################ GAME CLOSE TO END #################
    print("============ MINIMAX !!!!!!!")
    action = minimax(game, player, my_possible_moves) # Very little positions available, so rin minimax in a controlled breadth scenario
    if action == (-1,-1):
        return None
    return action
    ################ GAME CLOSE TO END #################

def make_move(game, player):
    """
    Make a movement with our recipe for success :)

    :param game: The game object
    :param player: Black (1) or White (2)

    :return: The best movement given the game object.
    """
    if player == 1:
        return make_move_black(game, player)
    else:
        return make_move_white(game, player)

def read():
    flag_new = True
    with open("input.txt", 'r') as f:
        player = int(f.readline().strip())
        previous_board = [[0 for x in range(5)] for y in range(5)]
        flag_new = True
        for i in range(5):
            line = f.readline().strip()
            for j in range(len(line)):
                previous_board[i][j] = int(line[j])
                if previous_board[i][j] != 0:
                    flag_new = False
                    

        board = [[0 for x in range(5)] for y in range(5)]
        for i in range(5):
            line = f.readline().strip()
            for j in range(len(line)):
                board[i][j] = int(line[j])

    f.close()
    """
    move = 1            
    if flag_new:
        if player == 1:
            file_counter = open("counter.txt", "w")
            file_counter.write("1")
            file_counter.close()
        else:
            file_counter = open("counter.txt", "w")
            file_counter.write("2")
            file_counter.close()
            move = 2
    else:
        with open("counter.txt", "r") as file_counter:
            move = int(file_counter.readline().strip()) + 2
        file_counter = open("counter.txt", "w")
        file_counter.write(str(move))
        file_counter.close()
    """       
    return player, previous_board, board


def write(action):
    print(str(action))
    file_write = open("output.txt", "w")
    if action == "PASS" or action is None:
        file_write.write("PASS")
    else:
        file_write.write("{},{}".format(action[0], action[1]))

    file_write.close()


player, previous_board, board = read()
#MAX_DEPTH = 5 if 24 - move > 5 else 24 - move
game = Board()
game.update_board(player, previous_board, board)
action = make_move(game, player)
write(action)








