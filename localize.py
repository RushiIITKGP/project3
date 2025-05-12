############################################################ DATA GENERATION #########################################
import numpy as np
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from heapq import heappush,heappop
import json
import os
import csv

with open("mazedata4900.json", "r") as file:
    maze_data = json.load(file)

######This function finds the neighbors of cell, neighbors can be top, bottom,right and left cell
def neighbours_list(x,y):
    neighbours=[]
    dir=[(0,1),(0,-1),(1,0),(-1,0)]
    for dx, dy in dir:
        ax, ay = x + dx, y + dy
        if 0 <= ax < D and 0 <= ay <D:
            neighbours.append([ax, ay])
    return neighbours

####open walls that  have only one open cells their neighbor#########
def openCells(graph,D,open_cells):
    while True:
        one_open_cell = []
        for i in range(1,D-1):
            for j in range(1,D-1):
                if graph[i, j] == "1":
                    one_open_neighbour = sum(graph[ix, iy] == "0" for ix, iy in neighbours_list(i, j))  #checking wall cell has how many open cells
                    #print(one_open_neighbour)
                    if one_open_neighbour == 1: ####@if wall cell has just one open cell as neighbor
                        one_open_cell.append((i, j)) ###then append hat wall cell to the list that we are maintaining to open cells randomly
                        #print(one_open_cell)

        if not one_open_cell:   ##if the list taht we are maintaining of wall cells that has only one  neighbor as open cell is empty, imples there rae no cells need gto be openend
            print("No more open cells")
            break

        i, j = random.choice(one_open_cell)  ##randomly select to open the wall cell that has only one open cell as neighbor
        if i!=0 and i!=D-1 and j!=0 and j!=D-1: ##if the randomly selected wall cell to open is not on the outer bounds(first, last row and first,last column)
            graph[i, j] = "0"  ####then open that cell
            open_cells.append((i, j))   ####append it to the list that we are mainitaining for total open cells that our maze has.
        else:
            graph[i,j] ="1"   ##if the randomly selected wall cell to open is in the outer bounds, donot open the wall cell
    return open_cells  ##at the end of full one iteration, return the list of all the cells that are open.


#############function to open the open cells that have only one open cell as neighbor
def deadCells(graph,cell_list):  ##cell_list contain the list of open cells in our maze)
    dead_cells = []
    for i, j in cell_list: ##chcek for each element in cell_list
        dead_cell_check = sum(graph[ix, iy] == "0" for ix, iy in neighbours_list(i, j)) ###if open cell has only one open neighbor then append that cell to the dead_cells list
        if dead_cell_check == 1:
            dead_cells.append([i, j])
    # print(len(dead_cells))

####randomly select half of the cells added to dead_cell list so that we do not open all the cells and make the maze too easy
    dead_cell_chosen = random.sample(dead_cells,len(dead_cells) // 2)

#####check for neighbors of open cells that  have been selected to having only one neighbor as open cell, open any one of the wall cell randomly if they are not in outer bound
    for i, j in dead_cell_chosen:
        valid_neighbour = [(mx, my) for mx, my in neighbours_list(i, j) if graph[mx, my] == "1"]
        # print(dead_cells_open)
        if valid_neighbour:
            mx,my=random.choice(valid_neighbour)
            if mx != 0 and mx != D-1 and my != 0 and my != D-1:
                graph[mx,my] ="0"
                cell_list.append([mx, my]) #####this list contains all dead cells that have been opened
            else:
                graph[mx,my]="1"
    return cell_list

def show_graph(graph,D,bot_pos,):
    plt.clf()  ##clears the graph and again plots it
    plt.imshow(graph=="1", cmap="binary",origin="lower",extent=[0,D,0,D])
    bx,by=bot_pos
    plt.scatter(by + 0.5, bx + 0.5, color="blue",label="bot")
    #grid should display D
    plt.xticks(ticks=np.arange(0,D+1,1))
    plt.yticks(ticks=np.arange(0,D+1,1))
    plt.title("rat catching")
    plt.legend(handlelength=0)
    plt.pause(10)

###manhattan distance heuristic
def heuristic(bot_pos,button_pos):
    bx,by=bot_pos
    bux,buy=button_pos
    return abs(bux-bx)+abs(buy-by)

def botMovement(graph,bot_pos,target,D):
    x,y=bot_pos
    gx,gy=target
    danger=[]
    directions=[(0,1),(0,-1),(1,0),(-1,0)]
    minheap=[]
    g_value={bot_pos:0}
    heappush(minheap,(0,0,(x,y),[(x,y)]))
    while minheap:
        f,g,(x,y),path=heappop(minheap)
        if (x,y)==(gx,gy):
            return path
        for i,j in directions:
            ix,iy=x+i,y+j
            if 0 <= ix < D and 0 <= iy < D and graph[ix, iy] == "0":
                node_g=g+1
                f=node_g+heuristic((ix, iy), target)
                if (ix,iy) not in g_value or node_g<g_value[ix,iy]:
                    g_value[ix,iy]=node_g
                    heappush(minheap,(f,node_g,(ix,iy),path+[(ix,iy)]))

    return "no path found from bot to button"


def get_moves_from_path(path):
    moves = []
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        if x2 == x1 + 1 and y2 == y1:
            moves.append('up')
        elif x2 == x1 - 1 and y2 == y1:
            moves.append('down')
        elif x2 == x1 and y2 == y1 + 1:
            moves.append('right')
        elif x2 == x1 and y2 == y1 - 1:
            moves.append('left')
        else:
            raise ValueError(f"Invalid step from {path[i]} to {path[i + 1]}")

    return moves


def update_belief(L, move, graph, D):
    L_new = set()

    #print("botpos -", bot_pos)
    #path = botMovement(graph, bot_est, target, D)
    #print("path - ", path)
    #moves = get_moves_from_path(path)
    #print("moves - ", moves)
    for lx,ly in L:
        pos = (lx,ly)

        x, y = pos
        if move == 'up' and x + 1 < D and graph[x + 1, y] == "0":
            pos = (x + 1, y)
            # print("bot moved up")
            #total_moves += 1
            # print("botpos -", bot_pos)
            # show_graph(graph, D, bot_pos)
        elif move == 'down' and x - 1 >= 0 and graph[x - 1, y] == "0":
            pos = (x - 1, y)
            # print("bot moved down")
            #total_moves += 1
            # print("botpos -", bot_pos)
            # show_graph(graph, D, bot_pos)
        elif move == 'right' and y + 1 < D and graph[x, y + 1] == "0":
            pos = (x, y + 1)
            # print("bot moved right")
            #total_moves += 1
            # print("botpos -", bot_pos)
            # show_graph(graph, D, bot_pos)
        elif move == 'left' and y - 1 >= 0 and graph[x, y - 1] == "0":
            pos = (x, y - 1)
            # print("bot moved left")
            #total_moves += 1
            # print("botpos -", bot_pos)
            # show_graph(graph, D, bot_pos)
        # else: print("bot stayed there")


        L_new.add(pos)

    return L_new


graph_dict = {}
#p=5644657
def simulation(result, L):
    graph_dict.clear()
    graph_dict[i] = result
    #graph_dict[p] = L
    #print(graph_dict)
    #print(graph_dict.items())

    #print(flat_list[3])
    save_data(graph_dict,L)


"""function to save data in csv file required for plotting the graph (q value vs result)"""
def save_data(graph_dict, L):
    file = "finalsteps.csv"
    file_present = os.path.exists(file)
    for i,j in graph_dict.items():
        selected = [i, j, L]

    with open(file, "a", newline="") as f:
        write = csv.writer(f)
        if not file_present:
            write.writerow(["|L|", "steps", "L"])
            print("CSV Prepared")
        write.writerow(selected)



if __name__=="__main__":
    D = 30
    #graph = np.full((D, D), "1")  # creates an array of 40*40 dimension and each has value 1, i.e there is wall
    #x, y = random.randrange(1, D - 2), random.randint(1, D - 2)  # randommly selects an x,y to open cell
    #graph[x, y] = "0"  # assigns value 0 to the open cell
    graph = np.array(maze_data[0]["maze"])
    #open_cells = [(x, y)]
    cell_list = []
    for i in range(1, D - 1):
        for j in range(1, D - 1):
            if graph[i][j] == '0':
                cell_list.append((i,j))

    bot_pos = random.choice(cell_list)
    bot_pos = tuple(bot_pos)
    print("initial_bot_pos =", bot_pos)
    #show_graph(graph, D, bot_pos)
    l = len(cell_list)
    #print("open cells  - ", cell_list)
    print("length - ", len(cell_list))

    dc = []
    for i, j in cell_list: ##chcek for each element in cell_list
        dead_cell_check = sum(graph[ix, iy] == "0" for ix, iy in neighbours_list(i, j)) ###if open cell has only one open neighbor then append that cell to the dead_cells list
        if dead_cell_check == 1:
            dc.append([i, j])

    corner_cells = []
    corner_coords = [(1, 1), (1, D - 2), (D - 2, 1), (D - 2, D - 2)]
    for x, y in corner_coords:
        if graph[x, y] == "0":
            corner_cells.append((x, y))

    dc_length = len(dc)
    # print("dead cells - ", dc)
    print("dc length - ", dc_length)

    dc.extend(corner_cells)

    dc_length = len(dc)
    #print("dead cells - ", dc)
    print("dc length - ", dc_length)

    for i in range(1, len(cell_list)+1):
        for p in range(100):
            print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk - ", i)
            L_store = []
            L = set(random.sample(cell_list, i))
            L = list(L)
            print("initial L -", L)
            L_store = list(L)

            total_moves = 0
            target = random.sample(dc, 1)
            target = tuple(target[0])

            print("target cell - ", target)

            #print("L - ", L)

            while (len(L) > 1) and total_moves<5000:

                bot_est = random.choice(L)
                bot_est = tuple(bot_est)
                print("bot_est -", bot_est)
                print("target - ", target)

                path = botMovement(graph, bot_est, target, D)
                # print("path - ", path)
                moves = get_moves_from_path(path)

                for move in moves:
                    L = update_belief(L, move, graph, D)
                    L = list(L)
                    total_moves += 1
                    #print("L_new - ", L)

                    # Update actual bot position
                    x, y = bot_pos
                    if move == 'up' and x + 1 < D and graph[x + 1,y] == "0":
                        bot_pos = (x + 1, y)
                        #print("graph - ", graph[x+1,y])
                    elif move == 'down' and x - 1 >= 0 and graph[x - 1,y] == "0":
                        bot_pos = (x - 1, y)
                    elif move == 'right' and y + 1 < D and graph[x,y + 1] == "0":
                        bot_pos = (x, y + 1)
                    elif move == 'left' and y - 1 >= 0 and graph[x,y - 1] == "0":
                        bot_pos = (x, y - 1)
                    else: bot_pos = (x,y)

                    #print("new bot pos -", bot_pos)

                    #if len(L) == 1:
                        #break

            print("L_final - ", L)
            print("total moves -", total_moves)
            simulation(total_moves, L_store)

    plt.show()







