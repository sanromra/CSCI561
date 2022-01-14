from queue import PriorityQueue
from collections import deque
import math
import heapq
def main():
    #file = "../resource/asnlib/public/sample/input6.txt" #Define the input file to read
    #file = "input.txt" #Define the input file to read
    file = "grading_test_cases/input1.txt"
    algorithm, len_x, len_y, len_z, in_x, in_y, in_z, out_x, out_y, out_z, grids, actions = read_input(file) #Read the input file
    
    """
    print("Starting point: " + str(in_x) + ", " + str(in_y) + ", " + str(in_z))
    print("End point: " + str(out_x) + ", " + str(out_y) + ", " + str(out_z))
    
    print("Grids -------------------------")
    for key, value in grids.items():
        print("    " + str(key) + " ==> " + str(value))
    
    print("Algorithm: " + algorithm)
    print("Lenght of maze: " + str(len_x) + ", " + str(len_y) + ", " + str(len_z))
    """
    
    path = None
    if algorithm.strip() == "BFS":
        path = BFS(in_x, in_y, in_z, out_x, out_y, out_z, grids, actions, len_x, len_y, len_z)
        file_write = open("output.txt", "w")
        if path is None:
            file_write.write("FAIL")
        else:
            file_write.write(str(len(path) - 1) + "\n")
            file_write.write(str(len(path)) + "\n")
            for i in range(len(path)):
                if i == 0:
                    line = str(path[i][0]) + " " + str(path[i][1]) + " " + str(path[i][2]) + " 0\n"
                elif i < len(path)-1:
                    line = str(path[i][0]) + " " + str(path[i][1]) + " " + str(path[i][2]) + " 1\n"
                else:
                    line = str(path[i][0]) + " " + str(path[i][1]) + " " + str(path[i][2]) + " 1"
                file_write.write(line)
        file_write.close()
                       
            
    
    elif algorithm.strip() == "UCS":
        path, cost = UCS(in_x, in_y, in_z, out_x, out_y, out_z, grids, len_x, len_y, len_z)
        file_write = open("output.txt", "w")
        if path is None:
            file_write.write("FAIL")
        else:
            file_write.write(str(cost) + "\n")
            file_write.write(str(len(path)) + "\n")
            for i in range(len(path)):
                if i < len(path)-1:
                    line = str(path[i][0]) + " " + str(path[i][1]) + " " + str(path[i][2]) + " " + str(path[i][3]) + "\n"
                else:
                    line = str(path[i][0]) + " " + str(path[i][1]) + " " + str(path[i][2]) + " " + str(path[i][3])
                file_write.write(line)
        file_write.close()
    elif algorithm.strip() == "A*":
        path, cost = A_star(in_x, in_y, in_z, out_x, out_y, out_z, grids, len_x, len_y, len_z)
        file_write = open("output.txt", "w")
        if path is None:
            file_write.write("FAIL")
        else:
            file_write.write(str(cost) + "\n")
            file_write.write(str(len(path)) + "\n")
            for i in range(len(path)):
                if i < len(path)-1:
                    line = str(path[i][0]) + " " + str(path[i][1]) + " " + str(path[i][2]) + " " + str(path[i][3]) + "\n"
                else:
                    line = str(path[i][0]) + " " + str(path[i][1]) + " " + str(path[i][2]) + " " + str(path[i][3])
                file_write.write(line)
        file_write.close()
    
    """
    print("Solution ------------------------------")
    if path is not None:
        for item in path:
            print(str(item))
    else:
        print("None")
    """
    
def read_input(input_file):
    with open(input_file, 'r') as file:
        algorithm = file.readline() #Read Algorithm string: either "BFS", "UCS" or "A*"
        len_x, len_y, len_z = [int(num) for num in file.readline().split()] #Size of each dimension
        in_x, in_y, in_z = [int(num) for num in file.readline().split()] #Starting point coordinates
        out_x, out_y, out_z = [int(num) for num in file.readline().split()] #Goal point coordinates
        n_grids = int(file.readline()) #Number of grids where actions are available
        grids = {}
        for i in range(n_grids):
            line = file.readline().split()
            grids[(int(line[0]), int(line[1]), int(line[2]))] = line[3:]
        file.close()

    actions = {
                '1':[(+1, 0, 0), 10],
                '2':[(-1, 0, 0), 10],
                '3':[(0, +1, 0), 10],
                '4':[(0, -1, 0), 10],
                '5':[(0, 0, +1), 10],
                '6':[(0, 0, -1), 10],
                '7':[(+1, +1, 0), 14],
                '8':[(+1, -1, 0), 14],
                '9':[(-1, +1, 0), 14],
                '10':[(-1, -1, 0), 14],
                '11':[(+1, 0, +1), 14],
                '12':[(+1, 0, -1), 14],
                '13':[(-1, 0, +1), 14],
                '14':[(-1, 0, -1), 14],
                '15':[(0, +1, +1), 14],
                '16':[(0, +1, -1), 14],
                '17':[(0, -1, +1), 14],
                '18':[(0, -1, -1), 14]
              }
    
    return algorithm, len_x, len_y, len_z, in_x, in_y, in_z, out_x, out_y, out_z, grids, actions

def take_action(x, y, z, n_action, search, out_x = None, out_y = None, out_z = None):
    
    if n_action == '1':
        if search == "BFS":
            return x+1, y, z, 1
        elif search == "UCS":
            return x+1, y, z, 10
        elif search == "A_star":
            return x+1, y, z, 10 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '2':
        if search == "BFS":
            return x-1, y, z, 1
        elif search == "UCS":
            return x-1, y, z, 10
        elif search == "A_star":
            return x-1, y, z, 10 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '3':
        if search == "BFS":
            return x, y+1, z, 1
        elif search == "UCS":
            return x, y+1, z, 10
        elif search == "A_star":
            return x, y+1, z, 10 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '4':
        if search == "BFS":
            return x, y-1, z, 1
        elif search == "UCS":
            return x, y-1, z, 10
        elif search == "A_star":
            return x, y-1, z, 10 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '5':
        if search == "BFS":
            return x, y, z+1, 1
        elif search == "UCS":
            return x, y, z+1, 10
        elif search == "A_star":
            return x, y, z+1, 10 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '6':
        if search == "BFS":
            return x, y, z-1, 1
        elif search == "UCS":
            return x, y, z-1, 10
        elif search == "A_star":
            return x, y, z-1, 10 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '7':
        if search == "BFS":
            return x+1, y+1, z, 1
        elif search == "UCS":
            return x+1, y+1, z, 14
        elif search == "A_star":
            return x+1, y+1, z, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '8':
        if search == "BFS":
            return x+1, y-1, z, 1
        elif search == "UCS":
            return x+1, y-1, z, 14
        elif search == "A_star":
            return x+1, y-1, z, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '9':
        if search == "BFS":
            return x-1, y+1, z, 1
        elif search == "UCS":
            return x-1, y+1, z, 14
        elif search == "A_star":
            return x-1, y+1, z, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '10':
        if search == "BFS":
            return x-1, y-1, z,1
        elif search == "UCS":
            return x-1, y-1, z,14
        elif search == "A_star":
            return x-1, y-1, z,14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '11':
        if search == "BFS":
            return x+1, y, z+1, 1
        elif search == "UCS":
            return x+1, y, z+1, 14
        elif search == "A_star":
            return x+1, y, z+1, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '12':
        if search == "BFS":
            return x+1, y, z-1, 1
        elif search == "UCS":
            return x+1, y, z-1, 14
        elif search == "A_star":
            return x+1, y, z-1, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '13':
        if search == "BFS":
            return x-1, y, z+1, 1
        elif search == "UCS":
            return x-1, y, z+1, 14
        elif search == "A_star":
            return x-1, y, z+1, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '14':
        if search == "BFS":
            return x-1, y, z-1, 1
        elif search == "UCS":
            return x-1, y, z-1, 14
        elif search == "A_star":
            return x-1, y, z-1, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '15':
        if search == "BFS":
            return x, y+1, z+1, 1
        elif search == "UCS":
            return x, y+1, z+1, 14
        elif search == "A_star":
            return x, y+1, z+1, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '16':
        if search == "BFS":
            return x, y+1, z-1, 1
        elif search == "UCS":
            return x, y+1, z-1, 14
        elif search == "A_star":
            return x, y+1, z-1, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '17':
        if search == "BFS":
            return x, y-1, z+1, 1
        elif search == "UCS":
            return x, y-1, z+1, 14
        elif search == "A_star":
            return x, y-1, z+1, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    elif n_action == '18':
        if search == "BFS":
            return x, y-1, z-1, 1
        elif search == "UCS":
            return x, y-1, z-1, 14
        elif search == "A_star":
            return x, y-1, z-1, 14 + heuristic(x, y, z, out_x, out_y, out_z)
    else:
        return None


def heuristic(x, y ,z, out_x, out_y, out_z):
    dif_x = out_x - x
    dif_y = out_y - y
    dif_z = out_z - z
    
    dif_x = dif_x * dif_x
    dif_y = dif_y * dif_y
    dif_z = dif_z * dif_z
    
    distance = math.sqrt(dif_x + dif_y + dif_z)
    
    return distance

def heuristic_1(x, y ,z, out_x, out_y, out_z):
    dif_x = abs(out_x - x)
    dif_y = abs(out_y - y)
    dif_z = abs(out_z - z)
    
    total_movements = dif_x + dif_y + dif_z
    
    total_points = (total_movements//2) * 14 + (total_movements%2) * 10
    
    return total_points
    
    
def check_validity(x, y , z, len_x, len_y, len_z):
    if x >= 0 and y >= 0 and z >= 0 and x < len_x and y < len_y and z < len_z:
        return True
    return False


def BFS(in_x, in_y, in_z, out_x, out_y, out_z, grids, actions, len_x, len_y, len_z):
    
    frontier = deque()
    frontier.append((in_x, in_y, in_z))
    explored = {}
    parent = {(in_x, in_y, in_z):None}

    while len(frontier) > 0:
        #path = frontier.pop(0) #Take the path from the frontier
        #print("Path: " + str(path))
        #node = path[-1] #Take the last node in the path, it is the one interesting to us
        node = frontier.popleft()
        #print("------ Node: " + str(node))
        """
        if node == (out_x, out_y, out_z):
            path = deque()
            while parent[node] is not None: 
                path.appendleft(node)
                #print(node)
                node = parent[node]
            path.appendleft(node)
            return path
        """
        actions = grids.get(node, [])
        #print("------------ Actions: " + str(actions))
        #print("------------ Intersection: " + str(len(explored.intersection({node}))))
        
        if explored.get(node, None) is None: #is not None and explored.get(node, None) is None:
            explored[node] = True
            #print("------------ Explored: ", explored)
            for action in actions:
                new_x, new_y, new_z, new_cost = take_action(node[0], node[1], node[2], action, "BFS")
                #print("------------------ Son: (" + str(new_x) + ", " + str(new_y) + ", " + str(new_z) + ")") 
                valid = check_validity(new_x, new_y, new_z, len_x, len_y, len_z)
                #print("------------------ Son is valid?: " + str(valid))
                if valid and explored.get((new_x, new_y, new_z), None) is None:
                    
                    if (new_x, new_y, new_z) not in parent:
                        parent[(new_x, new_y, new_z)] = node
                    
                    if (new_x, new_y, new_z) == (out_x, out_y, out_z):
                        path = deque()
                        node1 = (new_x, new_y, new_z)
                        while parent[node1] is not None: 
                            path.appendleft(node1)
                            #print(node)
                            node1 = parent[node1]
                        path.appendleft(node1)
                        return path
                    #print("Added: " + str((new_x, new_y, new_z)) + " ====> " + str(parent[(new_x, new_y, new_z)]))
                    #new_path = path + [(new_x, new_y, new_z)]
                    #print("------------------ New path: " + str(new_path))
                    frontier.append((new_x, new_y, new_z))
                    #print(frontier)
                    
                    
        #input("Continue")
                    
    return None
            
    
def UCS(in_x, in_y, in_z, out_x, out_y, out_z, grids, len_x, len_y, len_z):
    
    frontier = PriorityQueue() #Frontier as a priotity queue. Priority given by lowest cost path.
    frontier.put((0, (in_x, in_y, in_z))) #Introduce the initial point in the queue
    explored = {} #Explored set as a dictionary for improving accessing time
    parent = {(in_x, in_y, in_z):(None, 0)} #Keep track of the path without carrying it
    
    while frontier:
        path = frontier.get() #Take the path from the frontier. Tuple with cost of path and list of visited nodes.
        node = path[1] #Take the last node in the path, it is the one interesting to us.
        cost = path[0] #The cost of the taken path
        
        if (node[0], node[1], node[2]) == (out_x, out_y, out_z): 
            path = deque()
            node = (node[0], node[1], node[2])
            while parent[node][0] is not None: 
                path.appendleft((node[0], node[1], node[2], parent[node][1]))
                node = parent[node][0]
            path.appendleft((node[0], node[1], node[2], parent[node][1]))

            return path, cost #return path and cost if goal node reached
        
        actions = grids.get((node[0], node[1], node[2]), []) #See what actions are available to do at that maze point
        if explored.get((node[0], node[1], node[2]), None) is None: #actions is not None: #If actions available, meaning we can obtain children from the current node
            explored[(node[0], node[1], node[2])] = cost #Add the current node to the explored set to avoid loops
            for action in actions: #For each action
                new_x, new_y, new_z, new_cost = take_action(node[0], node[1], node[2], action, "UCS") #Obtain the correspondent child
                valid = check_validity(new_x, new_y, new_z, len_x, len_y, len_z) #Check if child is a point within the maze limits
                if valid and explored.get((new_x, new_y, new_z), None) is None: #If valid and not explored
                    parent[(new_x, new_y, new_z)] = (node, new_cost)  #Append new point to the path
                    frontier.put((cost+new_cost, (new_x, new_y, new_z))) #Put path in que queue alongside with its cost
                     #Pop the child node from path so that next child (if any) has its position available.
                    #print("------------------ Frontier: " + str(frontier))
                elif explored.get((new_x, new_y, new_z), None) is not None:# and explored[(new_x, new_y, new_z)] > cost+new_cost:
                    child_cost = explored[(new_x, new_y, new_z)]
                    if child_cost > cost + new_cost:
                        parent[(new_x, new_y, new_z)] = (node, new_cost)  #Append new point to the path
                        frontier.put((cost+new_cost, (new_x, new_y, new_z))) #Put path in que queue alongside with its cost
                
        #input("Continue")
                    
    return None, None

def A_star(in_x, in_y, in_z, out_x, out_y, out_z, grids, len_x, len_y, len_z):
    
    frontier = PriorityQueue()
    heuristic_points = heuristic(in_x, in_y, in_z, out_x, out_y, out_z)
    frontier.put((heuristic_points, 0, (in_x, in_y, in_z)))
    explored = {}
    costs = {(in_x, in_y, in_z):0}
    parent = {(in_x, in_y, in_z):(None, 0)}
    
    while frontier:
        path = frontier.get() #Take the path from the frontier
        #print("Path: " + str(path))
        node = path[-1] #Take the last node in the path, it is the one interesting to us
        cost = path[1]
        #print("------ Node: " + str(node))
        
        if (node[0], node[1], node[2]) == (out_x, out_y, out_z):
            path = deque()
            node = (node[0], node[1], node[2])
            while parent[node][0] is not None: 
                path.appendleft((node[0], node[1], node[2], parent[node][1]))
                node = parent[node][0]
            path.appendleft((node[0], node[1], node[2], parent[node][1]))

            return path, cost #return path and cost if goal node reached
        
        actions = grids.get((node[0], node[1], node[2]), [])
        #print("------------ Actions: " + str(actions))
        #print("------------ Intersection: " + str(len(explored.intersection({node}))))
        if explored.get((node[0], node[1], node[2]), None) is None:
            explored[(node[0], node[1], node[2])] = True
            #print("------------ Explored: ", explored)
            for action in actions:
                new_x, new_y, new_z, new_cost = take_action(node[0], node[1], node[2], action, "UCS")
                #print("------------------ Son: (" + str(new_x) + ", " + str(new_y) + ", " + str(new_z) + ")") 
                valid = check_validity(new_x, new_y, new_z, len_x, len_y, len_z)
                #print("------------------ Son is valid?: " + str(valid))
                if valid:# and explored.get((new_x, new_y, new_z), None) is None:
                    
                    child_cost = costs.get((new_x, new_y, new_z), math.inf)
                    #print("NODE: " + str((new_x, new_y, new_z)) + " has PREV COST: " + str(child_cost) + " and CURRENT COST: " + str(cost + new_cost))
                    if child_cost > cost + new_cost:
                        #print("--------- UPDATED")
                        #print("--------- Parent: " + str(node))
                    #print("------------------ New path: " + str(new_path))
                    #print("------------------ New cost: " + str(cost
                        g_x = cost+new_cost
                        f_x = g_x + heuristic(new_x, new_y, new_z, out_x, out_y, out_z)
                        parent[(new_x, new_y, new_z)] = (node, new_cost)
                        frontier.put((f_x, g_x, (new_x, new_y, new_z)))
                        costs[(new_x, new_y, new_z)] = g_x
                
                    
                    
        #input("Continue")
                    
    return None, None
    

main()

