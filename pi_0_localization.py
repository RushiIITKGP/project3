import csv
import numpy as np
from localization import update_belief, botMovement, get_moves_from_path
import pandas as pd
import json
import ast
import random

# CONFIG
D = 30
maze_file = "mazedata4900.json"
input_csv = "shared_L_bot_inputs.csv"
output_csv = "pi0_results.csv"

# Load maze
with open(maze_file, "r") as f:
    maze_data = json.load(f)
maze = np.array(maze_data[1]["maze"])

results = []

def neighbours_list(x,y):
    neighbours=[]
    dir=[(0,1),(0,-1),(1,0),(-1,0)]
    for dx, dy in dir:
        ax, ay = x + dx, y + dy
        if 0 <= ax < D and 0 <= ay <D:
            neighbours.append([ax, ay])
    return neighbours

with open(input_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        bot_pos = ast.literal_eval(row['bot_pos'])  # safely parse (x, y) tuple
        L_list = ast.literal_eval(row['L_subset'])  # safely parse list of (x, y) tuples
        L = set([tuple(pos) for pos in L_list])     # ensure it's a set of tuples

        original_L = list(L)                        # to store original L for output
        initial_L_size = len(original_L)

        total_moves = 0

        cell_list = []
        for i in range(1, D - 1):
            for j in range(1, D - 1):
                if maze[i,j] == '0':
                    cell_list.append((i, j))

        dc = []
        for i, j in cell_list:  ##chcek for each element in cell_list
            dead_cell_check = sum(maze[ix, iy] == "0" for ix, iy in neighbours_list(i,j))  ###if open cell has only one open neighbor then append that cell to the dead_cells list
            if dead_cell_check == 1:
                dc.append((i, j))

        corner_cells = []
        corner_coords = [(1, 1), (1, D - 2), (D - 2, 1), (D - 2, D - 2)]
        for x, y in corner_coords:
            if maze[x, y] == "0":
                corner_cells.append((x, y))
        dc.extend(corner_cells)
        target = random.choice(dc)


        while len(L) > 1 and total_moves < 5000:

            bot_est = random.choice(list(L))
            path = botMovement(maze, bot_est, target, D)
            moves = get_moves_from_path(path)

            for move in moves:
                L = update_belief(L, move, maze, D)
                total_moves += 1

                x, y = bot_pos
                if move == 'up' and x + 1 < D and maze[x + 1, y] == "0":
                    bot_pos = (x + 1, y)
                elif move == 'down' and x - 1 >= 0 and maze[x - 1, y] == "0":
                     bot_pos = (x - 1, y)
                elif move == 'right' and y + 1 < D and maze[x, y + 1] == "0":
                    bot_pos = (x, y + 1)
                elif move == 'left' and y - 1 >= 0 and maze[x, y - 1] == "0":
                    bot_pos = (x, y - 1)

        results.append({
            'L_size': initial_L_size,
            'L': str(original_L),
            'steps': total_moves
        })
        print(f"[Pi0] L_size={initial_L_size}, Moves={total_moves}")

# Save results
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"✅ Saved pi0 results to {output_csv}")
