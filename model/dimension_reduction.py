import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import mean_squared_error

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
df_solar = pd.read_csv("./solar/datasets_fill_NAN/fill_pro_Trina_23.4_87-Site_DKA-M9_A+C-Phases.csv")
df_solar = df_solar.iloc[:100000]  # Full dataset
print(df_solar.head())
print(df_solar.isnull().sum())

# Drop columns
df_solar.drop(["Wind_Speed", "Active_Energy_Delivered_Received", "Current_Phase_Average"], axis=1, inplace=True)
df_solar['timestamp'] = pd.to_datetime(df_solar['timestamp'])
df_solar = df_solar.set_index('timestamp')

# Features and target
X = df_solar.drop(['Active_Power'], axis=1)
y = df_solar[['Active_Power']]
print(X.head())

# Windowing function
def windowing(X_input, y_input, history_size):
    data, labels = [], []
    for i in range(history_size, len(y_input)):
        data.append(X_input[i - history_size:i, :])
        labels.append(y_input[i])
    return np.array(data), np.array(labels).reshape(-1, 1)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_shape):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape[-1], hidden_size=32, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32 * input_shape[-2], 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Training function for Autoencoder
def train_autoencoder(model, X_torch, epochs=50, batch_size=128, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_torch), batch_size):
            batch_X = X_torch[i:i + batch_size]
            optimizer.zero_grad()
            _, decoded = model(batch_X)
            loss = loss_fn(decoded, batch_X)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Autoencoder Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
    return model

# Training function for LSTM
@dataclass
class History:
    history_dict: dict

def train_model(model, X_train_torch, y_train_torch, X_val_torch, y_val_torch, epochs, batch_size, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    history_dict = {'loss': [], 'val_loss': []}
    
    for epoch_idx in range(epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, len(X_train_torch), batch_size):
            batch_X = X_train_torch[i:i + batch_size]
            batch_y = y_train_torch[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(X_train_torch)
        history_dict['loss'].append(train_loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_torch)
            val_loss = loss_fn(val_outputs, y_val_torch).item()
        history_dict['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch_idx + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return History(history_dict)

# Dimensionality reduction methods

def apply_PCA(X_input, n_components=None, cum_variance=None, if_apply=True):
    """
    应用PCA降维，支持指定主成分数量或累计方差贡献率。
    
    参数:
        X_input: 输入特征矩阵
        n_components: 指定的主成分数量（整数），优先于cum_variance
        cum_variance: 累计方差贡献率（如0.8），当n_components未指定时使用
        if_apply: 是否应用PCA，若为False则返回原数据
    
    返回:
        X_transformed: 降维后的特征矩阵
        n_components_used: 实际使用的输出维度
    """
    if not if_apply:
        return np.array(X_input), X_input.shape[1]
    
    # 初始化PCA
    if n_components is not None:
        # 确保n_components不超过特征数
        n_components = min(n_components, X_input.shape[1])
        pca = PCA(n_components=n_components)
    elif cum_variance is not None:
        pca = PCA(n_components=cum_variance)
    else:
        pca = PCA()  # 保留所有主成分
    
    # 创建Pipeline：标准化 + PCA
    scaler_pca = make_pipeline(MinMaxScaler(), pca)
    X_transformed = scaler_pca.fit_transform(X_input)
    
    # 获取实际使用的输出维度
    n_components_used = X_transformed.shape[1]
    
    return X_transformed, n_components_used

# def apply_PCA(X_input, cum_variance=0.8):
#     pca = PCA(n_components=cum_variance)
#     scaler_pca = make_pipeline(MinMaxScaler(), pca)
#     X_transformed = scaler_pca.fit_transform(X_input)
#     return X_transformed, pca.n_components_

def apply_KernelPCA(X_input, n_components):
    kpca = KernelPCA(n_components=n_components, kernel='rbf')
    scaler_kpca = make_pipeline(MinMaxScaler(), kpca)
    X_transformed = scaler_kpca.fit_transform(X_input)
    return X_transformed

def apply_Autoencoder(X_input, latent_dim, epochs=50, batch_size=128):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_input)
    X_torch = torch.FloatTensor(X_scaled).to(device)
    
    autoencoder = Autoencoder(input_dim=X_input.shape[1], latent_dim=latent_dim).to(device)
    autoencoder = train_autoencoder(autoencoder, X_torch, epochs, batch_size)
    
    autoencoder.eval()
    with torch.no_grad():
        X_encoded, _ = autoencoder(X_torch)
    return X_encoded.cpu().numpy()

def apply_RandomProjection(X_input, n_components):
    rp = GaussianRandomProjection(n_components=n_components)
    scaler_rp = make_pipeline(MinMaxScaler(), rp)
    X_transformed = scaler_rp.fit_transform(X_input)
    return X_transformed

# Comparison function with RMSE and plotting
def compare_methods(X, y, hist_size=24, epochs=50, batch_size=128):
    results = {}
    train_cutoff = int(0.7 * X.shape[0])
    val_cutoff = int(0.85 * X.shape[0])
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(y[:train_cutoff])
    y_norm = scaler_y.transform(y)
    
    methods = ['PCA', 'KernelPCA', 'Autoencoder', 'RandomProjection']
    
    # Initialize plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        start_time = time.time()
        
        # Apply dimensionality reduction
        if method == 'PCA':
            X_transformed, n_components = apply_PCA(X, cum_variance=0.9)
        # elif method == 'PCA_comp':
        #     X_transformed, n_components = apply_PCA(X, n_components = 6)
        elif method == 'KernelPCA':
            _, n_components = apply_PCA(X, cum_variance=0.9)
            X_transformed = apply_KernelPCA(X, n_components=n_components)
        elif method == 'Autoencoder':
            _, n_components = apply_PCA(X, cum_variance=0.9)
            X_transformed = apply_Autoencoder(X, latent_dim=n_components)
        elif method == 'RandomProjection':
            _, n_components = apply_PCA(X, cum_variance=0.9)
            X_transformed = apply_RandomProjection(X, n_components=n_components)
        
        print(f"{method} output shape: {X_transformed.shape}")
        
        # Prepare data for LSTM
        data_norm = np.concatenate((X_transformed, y_norm), axis=1)
        X_train, y_train = windowing(data_norm[:train_cutoff, :], data_norm[:train_cutoff, -1], hist_size)
        X_val, y_val = windowing(data_norm[train_cutoff:val_cutoff, :], data_norm[train_cutoff:val_cutoff, -1], hist_size)
        X_test, y_test = windowing(data_norm[val_cutoff:, :], data_norm[val_cutoff:, -1], hist_size)
        
        # Convert to PyTorch tensors
        X_train_torch = torch.FloatTensor(X_train).to(device)
        y_train_torch = torch.FloatTensor(y_train).to(device)
        X_val_torch = torch.FloatTensor(X_val).to(device)
        y_val_torch = torch.FloatTensor(y_val).to(device)
        X_test_torch = torch.FloatTensor(X_test).to(device)
        
        # Initialize and train LSTM
        model = LSTMModel(input_shape=X_train.shape).to(device)
        history = train_model(model, X_train_torch, y_train_torch, X_val_torch, y_val_torch, epochs, batch_size)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_torch)
            y_pred_inv = scaler_y.inverse_transform(y_pred.cpu().numpy())
            y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            test_mae = np.mean(np.abs(y_pred_inv - y_test_inv))
            test_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        
        # Calculate residuals for histogram
        residuals = y_pred_inv.flatten() - y_test_inv.flatten()
        
        # Plot error distribution
        sns.histplot(residuals, bins=30, kde=True, color='purple', ax=axes[idx])
        axes[idx].set_title(f'Error Distribution - {method}')
        axes[idx].set_xlabel('Prediction Error (Predicted - Actual)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(0, color='red', linestyle='--', label='Zero Error')
        axes[idx].legend()
        axes[idx].grid(True, linestyle='--', alpha=0.5)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        results[method] = {
            'val_loss': float(history.history_dict['val_loss'][-1]),
            'test_mae': float(test_mae),  # Ensure JSON serializable
            'test_rmse': float(test_rmse),
            'training_time': int(training_time),
            'n_components': int(n_components)
        }
        
        print(f"{method} - Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Training Time: {training_time:.2f}s")
    
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig('./solar/results/figures/error_distribution.png', dpi=300)
    plt.close()
    
    # Save results to JSON
    with open('./solar/results/datasets_info/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

# Run comparison
results = compare_methods(X, y, hist_size=24, epochs=50, batch_size=128)

# Print results
print("\nComparison Results:")
for method, metrics in results.items():
    print(f"{method}:")
    print(f"  Validation Loss: {metrics['val_loss']:.4f}")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Training Time: {metrics['training_time']:.2f}s")
    print(f"  Number of Components: {metrics['n_components']}")