import random
import sys
#from read import readInput
#from write import writeOutput

from GoBoard import Board

class GoRandomPlayer():
    def __init__(self):
        self.type = 'random'
        self.wb = None
    def set_wb(self, piece):
        self.wb = piece

    def get_input(self, go):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''        
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.is_valid_place(i, j, self.wb, test_check = True):
                    possible_placements.append((i,j))

        if not possible_placements:
            return "PASS"
        else:
            x, y = random.choice(possible_placements)
            flag = go.put_piece(x, y, self.wb)
            if self.wb == 1:
                print("BLACK MOVES: ({},{})".format(x, y))
            else:
                print("WHITE MOVES: ({},{})".format(x, y))
            return flag

    def move(self, go):
        return self.get_input(go)

    def learn(self, go):
        return None

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
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = GoRandomPlayer()
    action = player.get_input(go)
    writeOutput(action)