import joblib
import matplotlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import time
print(time.strftime("%H:%M:%S", time.localtime()))
matplotlib.rc("font", family='Microsoft YaHei')
electric_df = pd.read_excel('enhanced_data/cgan_electric_deepseek.xlsx', header=0)
emission_df = pd.read_excel('enhanced_data/cgan_emission_deepseek.xlsx', header=0)
assert electric_df.shape == emission_df.shape, "两个数据集形状不一致！"
X = electric_df.values.astype(np.float32)
y = emission_df.values.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
x_scaler = MinMaxScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)
class EnergyDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = EnergyDataset(X_train_scaled, y_train_scaled)
test_dataset = EnergyDataset(X_test_scaled, y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNEmissionsPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 16, 5, padding=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 11, 11),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cnn_layers(x)
        return self.fc(x.view(x.size(0), -1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNEmissionsPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
best_r2 = -float("inf")
train_history = {
    'train_loss': [],
    'test_mae': [],
    'test_mse': [],
    'test_r2': []
}
for epoch in range(2000):
    model.train()
    epoch_train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_train_loss += loss.item()

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    test_mae = mean_absolute_error(all_targets, all_preds)
    test_mse = mean_squared_error(all_targets, all_preds)
    test_r2 = r2_score(all_targets, all_preds)

    train_history['train_loss'].append(epoch_train_loss / len(train_loader))
    train_history['test_mae'].append(test_mae)
    train_history['test_mse'].append(test_mse)
    train_history['test_r2'].append(test_r2)

    if test_r2 > best_r2:
        best_r2 = test_r2
        torch.save(model.state_dict(), 'final_pretrained_model/best_cnn_model_CGAN.pth')
        print(f"Epoch {epoch:4d} | ★ 保存最佳模型 | Test R²: {test_r2:.4f}")

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:4d} | Train Loss: {epoch_train_loss / len(train_loader):.4f} | "
            f"Test MAE: {test_mae:.4f} | Test MSE: {test_mse:.4f} | Test R²: {test_r2:.4f}")
model.load_state_dict(torch.load('final_pretrained_model/best_cnn_model_CGAN.pth'))
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).cpu().numpy()
        all_preds.append(outputs)
        all_targets.append(targets.numpy())

y_pred = y_scaler.inverse_transform(np.concatenate(all_preds))
y_test = y_scaler.inverse_transform(np.concatenate(all_targets))

metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'R²': r2_score(y_test, y_pred),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred)
}

print("\n最终测试结果：")
for name, value in metrics.items():
    if name == 'MAPE':
        print(f"{name}: {value:.2%}")
    else:
        print(f"{name}: {value:.4f}")
plt.figure(figsize=(12, 6))
plt.plot(train_history['train_loss'], label='训练损失')
plt.plot(train_history['test_mae'], label='测试MAE')
plt.plot(train_history['test_mse'], label='测试MSE')
plt.xlabel('训练轮次')
plt.ylabel('指标值')
plt.title('训练过程监控')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
for i in range(11):
    plt.figure(figsize=(6, 4))

    indices = np.arange(len(y_test))

    plt.scatter(indices, y_test[:, i], color='blue', marker='o', label="真实值", alpha=0.6)
    plt.scatter(indices, y_pred[:, i], color='red', marker='x', label="预测值", alpha=0.6)

    plt.ylim(min(y_test[:, i].min(), y_pred[:, i].min()) - 0.1,
             max(y_test[:, i].max(), y_pred[:, i].max()) + 0.1)

    plt.xlabel("样本序号")
    plt.ylabel("取值")
    plt.title(electric_df.columns[i])

    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()
torch.save(model.state_dict(), 'final_model_result/pretrained_emission_predictor.pth')
joblib.dump(x_scaler, 'final_model_result/pretrained_x_scaler.pkl')
joblib.dump(y_scaler, 'final_model_result/pretrained_y_scaler.pkl')
print(time.strftime("%H:%M:%S", time.localtime()))