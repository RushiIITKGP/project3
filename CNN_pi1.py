import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import ast
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ---------- Dataset ----------
class LocalizationDataset(Dataset):
    def __init__(self, csv_file, json_map):
        df = pd.read_csv(csv_file)
        df = df[df['steps'] < 5000].reset_index(drop=True)

        with open(json_map, 'r') as f:
            maze_data = json.load(f)
        grid = np.array(maze_data[1]["maze"], dtype=np.float32)

        self.inputs, self.targets, self.L_sizes = [], [], []
        for _, row in df.iterrows():
            L_cells = ast.literal_eval(row['L'])
            steps = float(row['steps'])

            L_grid = np.zeros_like(grid, dtype=np.float32)
            for (x, y) in L_cells:
                L_grid[x, y] = 0.5

            input_array = np.stack([L_grid, grid], axis=0)
            self.inputs.append(torch.tensor(input_array * 2 - 1, dtype=torch.float32))
            self.targets.append(torch.tensor(steps, dtype=torch.float32))
            self.L_sizes.append(len(L_cells))

        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)
        self.L_sizes = torch.tensor(self.L_sizes)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.L_sizes[idx]

# ---------- CNN Model ----------
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
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_regressor = nn.Conv2d(64, 1, kernel_size=1)

        # self.fc1 = nn.Linear(128, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 1)

        # self.dropout1 = nn.Dropout(0.3)
        # self.dropout2 = nn.Dropout(0.3)
        # self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.global_pool(x)  # Shape: [B, 128, 1, 1]
        x = self.conv_regressor(x)  # Shape: [B, 1, 1, 1]
        x = F.softplus(x)  # Ensure non-negative output
        return x.view(x.size(0))

    # ---------- Training Script ----------
if __name__ == "__main__":
    csv_file = 'pi1_results.csv'
    json_file = 'mazedata4900.json'
    scale_factor = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LocalizationDataset(csv_file, json_file)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNN_MovePredictor().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 50
    best_r2 = -float('inf')
    patience = 10
    wait = 0

    train_losses = []


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_inputs, batch_targets, _ in pbar:
            batch_inputs = batch_inputs.to(device)
            batch_targets = (batch_targets / scale_factor).to(device)

            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = loss_function(preds, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Train Loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")



        # Validation pass
        model.eval()
        records = []
        with torch.no_grad():
            for batch_inputs, batch_targets, L_sizes in test_loader:
                batch_inputs = batch_inputs.to(device)
                preds = model(batch_inputs) * scale_factor
                preds = preds.detach().cpu().numpy()
                batch_targets = batch_targets.numpy()
                L_sizes = L_sizes.numpy()
                for l, p, a in zip(L_sizes, preds, batch_targets):
                    records.append({'L_size': l, 'Predicted': p, 'Actual': a})

        df_eval = pd.DataFrame(records)
        r2 = r2_score(df_eval['Actual'], df_eval['Predicted'])
        mae = mean_absolute_error(df_eval['Actual'], df_eval['Predicted'])
        mse = mean_squared_error(df_eval['Actual'], df_eval['Predicted'])
        rmse = mse ** 0.5

        scheduler.step(rmse)

        print(f"✅ R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        # ----------- Early Stopping Logic -----------
        if epoch > 15:
            if r2 > best_r2 + 1e-3:
                best_r2 = r2
                wait = 0
                df_eval.to_csv("best_predictions.csv", index=False)
                torch.save(model.state_dict(), "model_c1.pth")
                print(" Saved best model.")
            else:
                wait += 1
                if wait >= patience:
                    print(f"⏹ Early stopping triggered at epoch {epoch + 1}")
                    break

    print(f"\n✅ Training complete. Best R²: {best_r2:.4f}")


    # ---------- Smooth with Moving Average ----------
    def moving_average(values, window=5):
        return [sum(values[max(0, i - window + 1):i + 1]) / (i - max(0, i - window + 1) + 1) for i in
                range(len(values))]


    smoothed_train = moving_average(train_losses)

    # ---------- Plot ----------
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_train, label="Train Loss (Smoothed)")
    #plt.plot(smoothed_val, label="Validation RMSE (Smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    records = []

    for batch_inputs, batch_targets, batch_L_sizes in test_loader:
        batch_inputs = batch_inputs.to(device)
        preds = model(batch_inputs) * scale_factor  # rescale back

        preds = preds.detach().cpu().numpy()
        targets = batch_targets.numpy()
        L_sizes = batch_L_sizes.numpy()
        # print("Pred vs Actual (first batch):", list(zip(preds[:5], targets[:5])))

        for l_size, pred, actual in zip(L_sizes, preds, targets):
            records.append({
                'L_length': l_size,
                'Predicted': pred,
                'Actual': actual
            })
    # Create DataFrame
    df_eval = pd.DataFrame(records)

    # Calculate prediction error
    df_eval['Error'] = df_eval['Predicted'] - df_eval['Actual']



    df1 = pd.DataFrame(records)
    # print("records -", df1)

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
    plt.title('Comparison: Actual Pi1 Moves vs. CNN Model Prediction')
    plt.xlim(0, 550)
    plt.xticks(range(0, 551, 10), rotation=45, fontsize=10)
    plt.ylim(0, 550)
    plt.yticks(range(0, 550, 25))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.show()
