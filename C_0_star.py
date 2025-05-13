import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import json, ast, csv
from localization import update_belief

# Load maze
with open("mazedata4900.json", "r") as f:
    maze_data = json.load(f)
maze = np.array(maze_data[1]["maze"])
D = maze.shape[0]

def create_input_tensor(maze, L):
    # Match CNN.py format: wall = 1, open = 0
    maze_grid = (np.array(maze) == '1').astype(np.float32)  # 1 for wall, 0 for open
    L_grid = np.zeros_like(maze_grid, dtype=np.float32)
    for (x, y) in L:
        L_grid[x, y] = 0.5  # match CNN.py convention
    input_array = np.stack([L_grid, maze_grid], axis=0)
    return torch.tensor(input_array * 2 - 1, dtype=torch.float32)

class BellmanDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df[df['steps'] < 5000].reset_index(drop=True)

        with open("mazedata4900.json", 'r') as f:
            maze_data = json.load(f)
        grid = np.array(maze_data[1]["maze"], dtype=np.float32)

        self.inputs, self.L_vals = [], []
        for _, row in df.iterrows():
            L_cells = ast.literal_eval(row['L'])

            L_grid = np.zeros_like(grid, dtype=np.float32)
            for (x, y) in L_cells:
                L_grid[x, y] = 0.5

            input_array = np.stack([L_grid, (grid == 1).astype(np.float32)], axis=0)
            self.inputs.append(torch.tensor(input_array * 2 - 1, dtype=torch.float32))
            self.L_vals.append(set([tuple(x) for x in L_cells]))

        self.inputs = torch.stack(self.inputs)




    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        L = self.L_vals[idx]
        L_up = update_belief(L, 'up', maze, D)
        L_down = update_belief(L, 'down', maze, D)
        L_left = update_belief(L, 'left', maze, D)
        L_right = update_belief(L, 'right', maze, D)

        input_tensor = self.inputs[idx]
        L_dir_tensors = {
            'L': input_tensor,
            'L_up': create_input_tensor(maze, L_up),
            'L_down': create_input_tensor(maze, L_down),
            'L_left': create_input_tensor(maze, L_left),
            'L_right': create_input_tensor(maze, L_right),
        }
        return L_dir_tensors


class CNN_CStar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_regressor = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = self.conv_regressor(x)
        x = F.softplus(x)
        return x.view(x.size(0))

def bellman_loss(model, batch):
    gamma = 0.95  # Discount factor
    C_L = model(batch['L'])
    C_next = torch.stack([
        model(batch['L_up']),
        model(batch['L_down']),
        model(batch['L_left']),
        model(batch['L_right'])
    ], dim=1)
    C_min = torch.min(C_next, dim=1).values
    target = 1.0 + gamma * C_min  # Apply discounting
    return F.mse_loss(C_L, target)

if __name__ == '__main__':
    dataset = BellmanDataset('pi0_results.csv')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_CStar().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = bellman_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = bellman_loss(model, batch)
                test_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    torch.save(model.state_dict(), "model_c0_star.pth")
    print(" Trained C_0* model saved to model_c0_star.pth")

    # Evaluation for plotting
    model.eval()
    records = []
    # Convert belief grid from [-1, 1] back to [0, 1] and sum
    # Correctly compute L_sizes from each batch
    for batch in test_loader:
            L_tensor = batch['L'].to(device)
            preds = model(L_tensor).detach().cpu().numpy()
            L_sizes = [((tensor[0] + 1) / 2).sum().item() for tensor in batch['L']]
    with torch.no_grad():
        for batch in test_loader:
            L_tensor = batch['L'].to(device)
            preds = model(L_tensor).cpu().numpy()
              # Channel 0 is L belief
            actuals = 1 + torch.min(torch.stack([
                model(batch['L_up'].to(device)),
                model(batch['L_down'].to(device)),
                model(batch['L_left'].to(device)),
                model(batch['L_right'].to(device))
            ], dim=1), dim=1).values.detach().cpu().numpy()

            for l_size, pred, actual in zip(L_sizes, preds, actuals):
                l_size = round(l_size)
                records.append({
                    'L_length': round(l_size),
                    'Predicted': pred,
                    'Actual': actual
                })

    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(records)
    df_grouped = df.groupby('L_length').agg({'Predicted': 'mean', 'Actual': 'mean'}).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['L_length'], df_grouped['Actual'], 'o-', label='Actual (Bellman Target)')
    plt.plot(df_grouped['L_length'], df_grouped['Predicted'], 'x--', label='Predicted by C_0*')
    plt.xlabel('|L| - Initial Belief Set Size')
    plt.ylabel('Expected Steps to Localize')
    plt.title('C_0* Model: Predicted vs Actual by |L|')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Plot Loss Curve ----------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("C_0* Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
