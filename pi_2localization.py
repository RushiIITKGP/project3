
import csv
import numpy as np
import torch
from CNN_pi1 import (CNN_MovePredictor)
from localization import update_belief
from pi_1localization import run_pi1_strategy
import pandas as pd
import json
import ast
import random

# CONFIG
D = 30
maze_file = "mazedata4900.json"
input_csv = "shared_L_bot_inputs.csv"
output_csv = "pi2_results.csv"

# Load maze
with open(maze_file, "r") as f:
    maze_data = json.load(f)
maze = np.array(maze_data[1]["maze"])

def get_best_first_move_pi2(L, model, graph, D, device):
    directions = ['up', 'down', 'left', 'right']
    best_cost = float('inf')
    best_direction = None
    grid = (graph == '0').astype(np.float32)

    for direction in directions:
        L_next = update_belief(L, direction, graph, D)
        if not L_next:
            continue

        L_grid = np.zeros_like(grid)
        for x, y in L_next:
            L_grid[x, y] = 1.0

        input_array = np.stack([L_grid, grid], axis=0)
        input_tensor = torch.tensor(input_array * 2 - 1, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_cost = model(input_tensor).item() * 100

        if predicted_cost < best_cost:
            best_cost = predicted_cost
            best_direction = direction

    return best_direction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_MovePredictor().to(device)
model.load_state_dict(torch.load("model_c1.pth", map_location=device))
model.eval()

results = []

with open(input_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        bot_pos = ast.literal_eval(row['bot_pos'])
        L_list = ast.literal_eval(row['L_subset'])
        L = set([tuple(pos) for pos in L_list])
        original_L = list(L)
        initial_L_size = len(original_L)

        if initial_L_size == 1:
            results.append({
                'L_size': 1,
                'L': str(original_L),
                'steps': 0
            })
            print(f"[Pi2] L_size=1, already localized, Moves=0")
            continue

        total_moves = 0

        best_move = get_best_first_move_pi2(L, model, maze, D, device)
        if best_move:
            L = update_belief(L, best_move, maze, D)
            total_moves += 1

            x, y = bot_pos
            if best_move == 'up' and x + 1 < D and maze[x + 1, y] == "0":
                bot_pos = (x + 1, y)
            elif best_move == 'down' and x - 1 >= 0 and maze[x - 1, y] == "0":
                bot_pos = (x - 1, y)
            elif best_move == 'right' and y + 1 < D and maze[x, y + 1] == "0":
                bot_pos = (x, y + 1)
            elif best_move == 'left' and y - 1 >= 0 and maze[x, y - 1] == "0":
                bot_pos = (x, y - 1)

        total_moves += run_pi1_strategy(L, bot_pos, maze, D, model, device)

        results.append({
            'L_size': initial_L_size,
            'L': str(original_L),
            'steps': total_moves
        })
        print(f"[Pi2] L_size={initial_L_size}, Moves={total_moves}")

# Save results
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"✅ Saved pi2 results to {output_csv}")


