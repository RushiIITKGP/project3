import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import ast
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F

# ---------- Dataset ----------
class LocalizationDataset(Dataset):
    def __init__(self, csv_file, json_map):
        df = pd.read_csv(csv_file)
        df = df[df['steps'] < 5000].reset_index(drop=True)

        with open(json_map, 'r') as f:
            maze_data = json.load(f)
        grid = np.array(maze_data[10]["maze"], dtype=np.float32)

        self.inputs = []
        self.targets = []
        self.L_sizes = []

        for _, row in df.iterrows():
            L_cells = ast.literal_eval(row['L'])
            steps = float(row['steps'])

            L_grid = np.zeros_like(grid, dtype=np.float32)
            for (x, y) in L_cells:
                L_grid[x, y] = 1.0

            input_array = np.stack([L_grid, grid], axis=0)
            input_tensor = torch.tensor(input_array * 2 - 1, dtype=torch.float32)
            target_tensor = torch.tensor(steps, dtype=torch.float32)

            self.inputs.append(input_tensor)
            self.targets.append(target_tensor)
            self.L_sizes.append(len(L_cells))

        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)
        self.L_sizes = torch.tensor(self.L_sizes)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.L_sizes[idx]

# ---------- Model ----------
class CNN_MovePredictor(nn.Module):
    def __init__(self):
        super(CNN_MovePredictor, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(4 * 4 * 64, 128)
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.linear1(x)))
        return self.linear2(x).squeeze(1)

# ---------- Main ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_factor = 100

    dataset = LocalizationDataset('finalsteps.csv', 'mazedata4900.json')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNN_MovePredictor().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # ---------- Train ----------
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch_inputs, batch_targets, _ in pbar:
            batch_inputs = batch_inputs.to(device)
            batch_targets = (batch_targets / scale_factor).to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Batch Loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Train Loss: {avg_loss:.4f}")

    # ---------- Predict on test data ----------
    records = []

    for batch_inputs, batch_targets, batch_L_sizes in test_loader:
        batch_inputs = batch_inputs.to(device)
        preds = model(batch_inputs) * scale_factor  # rescale back

        preds = preds.detach().cpu().numpy()
        targets = batch_targets.numpy()
        L_sizes = batch_L_sizes.numpy()

        for l_size, pred, actual in zip(L_sizes, preds, targets):
            records.append({
                'L_length': l_size,
                'Predicted': pred,
                'Actual': actual
            })

    # ---------- Save to CSV ----------
    df_out = pd.DataFrame(records)
    df_out.to_csv('predictions_on_test.csv', index=False)
    print("✅ Saved predictions to predictions_on_test.csv")
