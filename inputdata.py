import random
import json
import csv
import numpy as np

# CONFIG
D = 30
num_trials = 50
maze_file = "mazedata4900.json"
output_csv = "shared_L_bot_inputs.csv"

# Load maze data
with open(maze_file, "r") as file:
    maze_data = json.load(file)

# Use maze 0
maze = np.array(maze_data[1]["maze"])

# Find open cells
cell_list = [(i, j) for i in range(1, D - 1) for j in range(1, D - 1) if maze[i][j] == '0']

# Fixed random seed for reproducibility
random.seed(123)

# Sample shared bot position
bot_pos = random.choice(cell_list)

# Open CSV file for writing
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['bot_pos', 'L_subset'])

    for i in range(1, len(cell_list) + 1):
        for p in range(num_trials):
            L_sample = random.sample(cell_list, i)
            L_list_string = str(L_sample)
            bot_string = f"({bot_pos[0]}, {bot_pos[1]})"
            writer.writerow([bot_string, L_list_string])

print(f" Saved shared bot positions and L subsets to {output_csv}")
