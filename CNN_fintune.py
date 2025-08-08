import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib
import time

print(time.strftime("%H:%M:%S", time.localtime()))
matplotlib.rc("font", family='Microsoft YaHei')


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

    def forward(self, x):
        x = self.cnn_layers(x)
        return self.fc(x.view(x.size(0), -1))


class EnergyDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


print("加载原始数据...")
electric_data = pd.read_excel('data/electric_data.xlsx')
emission_data = pd.read_excel('data/emission_data.xlsx')

emission_data = emission_data.dropna(axis=1, how='all')
if 'Unnamed: 11' in emission_data.columns:
    emission_data = emission_data.drop('Unnamed: 11', axis=1)

print(f"电力数据形状: {electric_data.shape}")
print(f"排放数据形状: {emission_data.shape}")

min_samples = min(len(electric_data), len(emission_data))
electric_data = electric_data.iloc[:min_samples]
emission_data = emission_data.iloc[:min_samples]

X = electric_data.values.astype(np.float32)
y = emission_data.values.astype(np.float32)

print(f"处理后 - 电力数据形状: {X.shape}")
print(f"处理后 - 排放数据形状: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

x_scaler = MinMaxScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

train_dataset = EnergyDataset(X_train_scaled, y_train_scaled)
test_dataset = EnergyDataset(X_test_scaled, y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = CNNEmissionsPredictor().to(device)
model.load_state_dict(torch.load('final_model_result/pretrained_emission_predictor.pth', map_location=device))
print("成功加载预训练模型")

print("开始微调模型...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

num_epochs = 1000
best_test_loss = float('inf')
early_stop_counter = 0
early_stop_patience = 100
best_r2 = -float('inf')
history = {
    'train_loss': [],
    'test_loss': [],
    'test_metrics': {
        'MSE': [], 'MAE': [], 'R²': [], 'MAPE': []
    }
}
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            test_loss += criterion(outputs.cpu(), targets).item()

    avg_test_loss = test_loss / len(test_loader)
    history['test_loss'].append(avg_test_loss)

    y_pred_scaled = np.concatenate(all_preds)
    y_true_scaled = np.concatenate(all_targets)

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

    history['test_metrics']['MSE'].append(mse)
    history['test_metrics']['MAE'].append(mae)
    history['test_metrics']['R²'].append(r2)
    history['test_metrics']['MAPE'].append(mape)

    scheduler.step(avg_test_loss)

    if r2 > best_r2:
        best_r2 = r2
        torch.save(model.state_dict(), 'final_finetune_model/best_cnn_model_CGAN.pth')
        early_stop_counter = 0
        print(f"Epoch {epoch + 1}/{num_epochs} | ★ 保存最佳模型 | R²: {r2:.4f}")
    else:
        early_stop_counter += 1

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f} | "
              f"MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")

    if early_stop_counter >= early_stop_patience:
        print(f"早停: {early_stop_patience}轮内R²没有提升，在第{epoch + 1}轮停止训练")
        break

print("\n加载最佳微调模型进行最终评估...")
model.load_state_dict(torch.load('final_finetune_model/best_cnn_model_CGAN.pth'))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())

y_pred_scaled = np.concatenate(all_preds)
y_true_scaled = np.concatenate(all_targets)

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_true_scaled)

final_metrics = {
    'MSE': mean_squared_error(y_true, y_pred),
    'MAE': mean_absolute_error(y_true, y_pred),
    'R²': r2_score(y_true, y_pred),
    'MAPE': np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
}

print("\n=== 微调后的模型在原始数据上的评估结果 ===")
for name, value in final_metrics.items():
    if name == 'MAPE':
        print(f"{name}: {value:.2f}%")
    else:
        print(f"{name}: {value:.4f}")

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['test_loss'], label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练/测试损失')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 2)
    plt.plot(history['test_metrics']['R²'], label='R²', color='green')
    plt.axhline(y=best_r2, color='r', linestyle='--', label=f'最佳R²: {best_r2:.4f}')
    plt.xlabel('轮次')
    plt.ylabel('R²值')
    plt.title('测试集R²变化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 3)
    plt.plot(history['test_metrics']['MSE'], label='MSE', color='red')
    plt.plot(history['test_metrics']['MAE'], label='MAE', color='blue')
    plt.xlabel('轮次')
    plt.ylabel('误差值')
    plt.title('MSE和MAE变化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('final_result/微调过程.png')
    plt.show()

    print("训练历史图表已保存为 'final_result/微调过程.png'")
except Exception as e:
    print(f"绘图失败: {e}，但不影响模型训练和评估")

torch.save(model.state_dict(), 'final_model_result/finetune_emission_predictor.pth')
joblib.dump(x_scaler, 'final_model_result/finetune_x_scaler.pkl')
joblib.dump(y_scaler, 'final_model_result/finetune_y_scaler.pkl')

print(time.strftime("%H:%M:%S", time.localtime()))