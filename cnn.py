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
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------- Dataset ----------
class LocalizationDataset(Dataset):
    def __init__(self, csv_file, json_map):
        df = pd.read_csv(csv_file)
        df = df[df['steps'] < 5000].reset_index(drop=True)

        with open(json_map, 'r') as f:
            maze_data = json.load(f)
        grid = np.array(maze_data[0]["maze"], dtype=np.float32)

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

        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # flatten

        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return F.relu(self.fc3(x)).squeeze(1)

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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

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

    df1 = pd.DataFrame(records)

    # Group by L_length and compute averages
    grouped = df1.groupby('L_length').agg({
        'Predicted': 'mean',
        'Actual': 'mean'
    }).reset_index()

    # Plot
    plt.figure(figsize=(16, 5))
    plt.plot(grouped['L_length'], grouped['Actual'], 'o-', label='Actual Avg Moves (Pi0 Simulation)')
    plt.plot(grouped['L_length'], grouped['Predicted'], 'x--', label='Predicted Avg Moves (CNN Model)')

    plt.xlabel('Initial Belief Set Size |L|')
    plt.ylabel('Average Moves to Localize')
    plt.title('Comparison: Actual Pi0 Moves vs. CNN Model Prediction')
    plt.xlim(0, 550)
    plt.xticks(range(0, 551, 10), rotation=45, fontsize=10)
    plt.ylim(0, 450)
    plt.yticks(range(0, 450, 25))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()


    plt.show()


    # ---------- Compute R² ----------
    all_preds = np.array([r['Predicted'] for r in records])
    all_actuals = np.array([r['Actual'] for r in records])

    r2 = r2_score(all_actuals, all_preds)
    mae = mean_absolute_error(all_actuals, all_preds)
    mse = mean_squared_error(all_actuals, all_preds)
    rmse = np.sqrt(mse)

    print(f"✅ R²: {r2:.4f}")
    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ RMSE: {rmse:.4f}")

    # ---------- Save to CSV ----------
    df_out = pd.DataFrame(records)
    df_out.to_csv('predictions_on_test.csv', index=False)
    print("✅ Saved predictions to predictions_on_test.csv")
