import torch
import csv
import ast
import json
import numpy as np
from C_0_star import CNN_CStar
from localization import update_belief

# ---------- Config ----------
D = 30
MAX_STEPS = 500
GAMMA = 0.95
maze_file = "mazedata4900.json"
input_csv = "shared_L_bot_inputs.csv"
output_csv = "new_pi1_results.csv"
model_path = "model_c0_star.pth"

# ---------- Load maze ----------
with open(maze_file, "r") as f:
    maze_data = json.load(f)
maze = np.array(maze_data[1]["maze"])


def create_input_tensor(maze, L):
    maze_grid = (np.array(maze) == '1').astype(np.float32)  # 1 for wall, 0 for open
    L_grid = np.zeros_like(maze_grid, dtype=np.float32)
    for (x, y) in L:
        L_grid[x, y] = 0.5
    input_array = np.stack([L_grid, maze_grid], axis=0)
    return torch.tensor(input_array * 2 - 1, dtype=torch.float32)


# ---------- Load C_0* Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_CStar().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- π₁ Policy Evaluation ----------
results = []

with open(input_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        bot_pos = ast.literal_eval(row['bot_pos'])
        L_list = ast.literal_eval(row['L_subset'])
        grid = np.array(maze_data[1]["maze"], dtype=np.float32)
        L = set([tuple(pos) for pos in L_list])
        initial_size = len(L)

        if initial_size <= 1:
            results.append({'L_size': initial_size, 'L': str(L_list), 'steps': 0})
            continue

        steps = 0
        current_L = L
        print("Initial L -", current_L)

        while len(current_L) > 1 and steps < MAX_STEPS:
            best_score = float('inf')
            best_move = None

            print(f"\nStep {steps}, |L| = {len(current_L)}")
            for direction in ['up', 'down', 'left', 'right']:
                L_next = update_belief(current_L, direction, maze, D)
                if not L_next:
                    print(f"  {direction}: invalid (empty L)")
                    continue

                L_grid = np.zeros_like(grid)
                for x, y in L_next:
                    L_grid[x, y] = 0.5

                input_array = np.stack([L_grid, grid], axis=0)
                input_tensor = torch.tensor(input_array * 2 - 1, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    predicted_cost = model(input_tensor).item()
                total_cost = 1 + GAMMA * predicted_cost

                print(f"  {direction}: predicted cost = {total_cost:.2f}, |L'| = {len(L_next)}")

                # Select best move
                if total_cost < best_score:
                    best_score = total_cost
                    best_move = direction
                    best_L = L_next


            if best_move is None:
                break  # No valid moves

            current_L = best_L
            #print("L_next -", current_L)
            steps += 1

            x, y = bot_pos
            if best_move == 'up' and x + 1 < D and maze[x + 1, y] == "0":
                bot_pos = (x + 1, y)
            elif best_move == 'down' and x - 1 >= 0 and maze[x - 1, y] == "0":
                bot_pos = (x - 1, y)
            elif best_move == 'right' and y + 1 < D and maze[x, y + 1] == "0":
                bot_pos = (x, y + 1)
            elif best_move == 'left' and y - 1 >= 0 and maze[x, y - 1] == "0":
                bot_pos = (x, y - 1)

        results.append({'L_size': initial_size, 'L': str(L_list), 'steps': steps})
        print(f"[π₁] L_size={initial_size}, Steps={steps}")

# ---------- Save Results ----------
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['L_size', 'L', 'steps'])
    writer.writeheader()
    writer.writerows(results)

print(f" Saved π₁ Bellman results to {output_csv}")
