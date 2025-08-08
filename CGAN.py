import matplotlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import time
print(time.strftime("%H:%M:%S", time.localtime()))

matplotlib.rc("font", family='Microsoft YaHei')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

electric = pd.read_excel('data/electric_data.xlsx', sheet_name='Sheet1')
emission = pd.read_excel('data/emission_data.xlsx', sheet_name='Sheet1')

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(electric)
y = scaler_y.fit_transform(emission)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)


class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, noise, conditions):
        x = torch.cat([noise, conditions], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim + feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, conditions):
        x = torch.cat([inputs, conditions], dim=1)
        return self.model(x)


latent_dim = 32
batch_size = 32
epochs = 5000
lr = 0.0002
beta1 = 0.5

generator = Generator(latent_dim, X.shape[1], y.shape[1]).to(device)
discriminator = Discriminator(X.shape[1], y.shape[1]).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, (real_conditions, real_emissions) in enumerate(dataloader):
        batch_size = real_conditions.size(0)

        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_emissions = generator(noise, real_conditions)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        outputs_real = discriminator(real_emissions, real_conditions)
        loss_real = criterion(outputs_real, real_labels)

        outputs_fake = discriminator(fake_emissions.detach(), real_conditions)
        loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = loss_real + loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        outputs = discriminator(fake_emissions, real_conditions)
        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()


def generate_samples(num_samples, conditions):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        conditions = torch.tensor(conditions, dtype=torch.float32).to(device)
        generated = generator(noise, conditions)
    return scaler_y.inverse_transform(generated.cpu().numpy())


x_new = scaler_x.transform(pd.concat([electric] * 20, ignore_index=True))[:1000]
y_new = generate_samples(1000, x_new)

y_new = np.clip(y_new, a_min=0, a_max=None)


def evaluate_generated(real_data, fake_data):
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_data)
    fake_pca = pca.transform(fake_data)

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real')
    plt.title('Real Data Distribution')

    plt.subplot(122)
    plt.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5, color='r', label='Fake')
    plt.title('Generated Data Distribution')
    plt.show()

    ks_results = []
    for i in range(real_data.shape[1]):
        stat, p = ks_2samp(real_data[:, i], fake_data[:, i])
        ks_results.append((stat, p))

    print(f"KS Test Results (mean p-value: {np.mean([x[1] for x in ks_results]):.4f})")

    X_combined = np.vstack([real_data, fake_data])
    y_combined = np.hstack([np.zeros(real_data.shape[0]), np.ones(fake_data.shape[0])])

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3)

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(real_data.shape[1], 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    clf = Classifier().to(device)
    criterion_clf = nn.BCELoss()
    optimizer_clf = optim.Adam(clf.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    for _ in range(50):
        outputs = clf(X_train_tensor)
        loss = criterion_clf(outputs, y_train_tensor)
        optimizer_clf.zero_grad()
        loss.backward()
        optimizer_clf.step()

    with torch.no_grad():
        test_outputs = clf(X_test_tensor)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

    print(f"Discriminator Accuracy: {accuracy:.4f} (接近0.5表示数据质量高)")


evaluate_generated(emission.values, y_new)

pd.DataFrame(scaler_x.inverse_transform(x_new), columns=electric.columns).to_excel('enhanced_data/cgan_electric_deepseek.xlsx',
                                                                                   index=False)
pd.DataFrame(y_new, columns=emission.columns).to_excel('enhanced_data/cgan_emission_deepseek.xlsx', index=False)

print(time.strftime("%H:%M:%S", time.localtime()))
