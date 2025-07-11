import numpy as np
import pandas as pd
import math
from datetime import datetime
import seaborn as sns
import xgboost as xgb
import gc
import json
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# PCA application   
def apply_PCA(X_input, cum_variance, if_apply):
    if if_apply:
        pca = PCA(n_components=cum_variance)
        scaler_pca = make_pipeline(MinMaxScaler(), pca)
        return scaler_pca.fit_transform(X_input)
    return np.array(X_input)

# Windowing for time series
def windowing(X_input, y_input, history_size):
    data, labels = [], []
    for i in range(history_size, len(y_input)):
        data.append(X_input[i - history_size:i, :])
        labels.append(y_input[i])
    return np.array(data), np.array(labels).reshape(-1, 1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return output
    
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(attention_output)
        return output

class CNNAttentionLSTMModel(nn.Module):
    def __init__(self, input_shape, num_heads=4, out_channels=64, kernel_size=3, hidden_size = 256):
        super(CNNAttentionLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[-1], out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.attention = MultiHeadAttention(d_model=out_channels, num_heads=num_heads)
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(8 * input_shape[-2], hidden_size)
        self.dense2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # 输入形状: (batch, time, features)
        x = x.permute(0, 2, 1)  # (batch, features, time)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (batch, time, features)
        # 添加注意力机制
        x = self.attention(x)
        # LSTM处理
        x, _ = self.lstm(x)
        # 展平并通过全连接层
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_shape, hidden_size = 8, hidden_size2 = 128, num_layers = 1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(hidden_size * input_shape[-2], hidden_size2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(hidden_size2, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

# History class for compatibility
class History:
    def __init__(self, history_dict):
        self.history = history_dict

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, device, patience=10):
    """
    训练模型并返回历史记录和最佳验证损失
    Args:
        model: 模型实例
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 设备
        patience: 早停耐心值
    Returns:
        history: 训练和验证损失历史
        best_val_loss: 最佳验证MAE
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()  # MAE 损失
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # 验证（分批处理以节省内存）
        model.eval()
        val_loss = 0
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)  # 验证批次可较大
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += loss_fn(outputs, y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 清空内存
        torch.cuda.empty_cache()
    
    return history, best_val_loss

# Optuna 目标函数：LSTMModel
def objective_lstm(trial):
    # 定义超参数搜索空间
    hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 16, 32])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_size2 = trial.suggest_categorical('hidden_size2', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # 初始化模型
    model = LSTMModel(
        input_shape=X_train.shape[-2:], 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        hidden_size2=hidden_size2
    ).to(device)
    
    # 修改模型的 dropout 层
    model.dropout = nn.Dropout(dropout_rate)
    
    # 训练模型
    history, val_loss = train_model(
        model, X_train_torch, y_train_torch, X_val_torch, y_val_torch, 
        epochs=50, batch_size=batch_size, learning_rate=learning_rate, device=device
    )
    
    # 释放内存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_loss

# Optuna 目标函数：CNNAttentionLSTMModel
def objective_cnn_attention_lstm(trial):
    # 定义超参数搜索空间
    out_channels = trial.suggest_categorical('out_channels', [32, 64, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 16])
    hidden_size2 = trial.suggest_categorical('hidden_size2', [128, 256, 512])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # 确保 out_channels 能被 num_heads 整除
    if out_channels % num_heads != 0:
        raise optuna.TrialPruned()
    
    # 获取 input_shape
    time_steps = X_train.shape[-2]  # 固定时间步数
    features = X_train.shape[-1]
    input_shape = (time_steps, features)
    
    # 检查 dense1 输入维度
    expected_dense1_input = hidden_size * time_steps
    print(f"Trial {trial.number}: hidden_size={hidden_size}, time_steps={time_steps}, expected_dense1_input={expected_dense1_input}")
    
    # 初始化模型
    model = CNNAttentionLSTMModel(
        input_shape=input_shape,
        num_heads=num_heads,
        out_channels=out_channels,
        kernel_size=kernel_size,
        hidden_size=hidden_size,
        hidden_size2=hidden_size2
    ).to(device)
    
    # 验证 dense1 维度
    if model.dense1.in_features != expected_dense1_input:
        print(f"Warning: dense1 input dimension mismatch. Expected {expected_dense1_input}, got {model.dense1.in_features}")
        raise optuna.TrialPruned()
    
    # 训练模型
    history, val_loss = train_model(
        model, X_train_torch, y_train_torch, X_val_torch, y_val_torch,
        epochs=50, batch_size=batch_size, learning_rate=learning_rate, device=device
    )
    
    # 释放内存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_loss

# 优化超参数
def optimize_hyperparameters(model_type, n_trials=50):
    """
    使用 Optuna 进行超参数优化，并保存每次试验的超参数和目标值到 JSON
    Args:
        model_type: 'lstm' 或 'cnn_attention_lstm'
        n_trials: 优化试验次数
        json_path: 保存试验结果的 JSON 文件路径
    Returns:
        best_params: 最优超参数
        best_value: 最优验证损失
    """
    # 初始化试验记录列表
    trials_history = []
    json_path=f'./solar/results/hyperparameter_trials_{model_type}.json'
    # 定义回调函数，记录每次试验
    def callback(study, trial):
        trial_data = {
            'trial_number': trial.number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            'params': trial.params,
            'value': trial.value,  # 目标值（验证集 MAE）
            'state': str(trial.state)  # 试验状态（如 COMPLETED, PRUNED）
        }
        trials_history.append(trial_data)
    
    # 创建 Optuna 优化任务
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # 运行优化
    if model_type == 'lstm':
        study.optimize(objective_lstm, n_trials=n_trials, callbacks=[callback])
    elif model_type == 'cnn_attention_lstm':
        study.optimize(objective_cnn_attention_lstm, n_trials=n_trials, callbacks=[callback])
    else:
        raise ValueError("model_type must be 'lstm' or 'cnn_attention_lstm'")
    
    # 保存试验记录到 JSON
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(trials_history, f, indent=4, ensure_ascii=False)
    
    # 获取最优结果
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best parameters for {model_type}: {best_params}")
    print(f"Best validation MAE: {best_value:.3f}")
    print(f"Trial history saved to {json_path}")
    
    return best_params, best_value

def evaluate_model(model, X_test_torch, y_test, y_test_inv, scaler_y, model_name):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_torch).cpu().numpy()
    print('\n\n---------------------------------------------------')
    print(f'{model_name} MAE for test set: {round(mean_absolute_error(y_pred, y_test), 3)}')
    print('---------------------------------------------------')
    
    y_pred_actual = scaler_y.inverse_transform(y_pred)
    os.makedirs('./solar/results/result_data', exist_ok=True)
    results_df = pd.DataFrame({'Actual': y_test_inv.flatten(), 'Predicted': y_pred_actual.flatten()})
    results_df.to_csv(f'./solar/results/result_data/{model_name}_actual_predict.csv', index=False)
    return y_pred_actual

# Plotting function
def plot_results(y_pred_actual, y_test_inv, model, model_name, evals_result=None):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].plot(y_pred_actual[:1000], label='Prediction')
    ax[0].plot(y_test_inv[:1000], label='Actual')
    ax[0].legend()
    ax[0].set_title(f'Prediction vs Actual (First 1000) - {model_name}')
    ax[0].set_xlabel('Observation')
    ax[0].set_ylabel('Price')
    
    if model_name == 'XGBoost' and evals_result is not None:
        try:
            train_mae = evals_result['train']['mae']
            val_mae = evals_result['val']['mae']
            ax[1].plot(train_mae, label='Training MAE')
            ax[1].plot(val_mae, label='Validation MAE')
        except KeyError:
            print("Error: evals_result missing 'mae' key. Check if eval_metric='mae' is set.")
            print("Current evals_result keys:", evals_result.keys())
        ax[1].set_title(f'Training and Validation MAE - {model_name}')
    else:
        ax[1].plot(model.history['loss'], label='Training Loss')
        ax[1].plot(model.history['val_loss'], label='Validation Loss')
        ax[1].legend()
        ax[1].set_title(f'Training and Validation MAE ({model_name})')
    ax[1].set_xlabel('Iteration/Epochs')
    ax[1].set_ylabel('MAE')
    
    fig.tight_layout()
    os.makedirs('./solar/results/figures', exist_ok=True)
    plt.savefig(f"./solar/results/figures/{model_name}.png")
    plt.close()

# 训练和评估最优模型
def train_and_evaluate_best_model(model_type, best_params):
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    
    if model_type == 'lstm':
        model = LSTMModel(
            input_shape=X_train.shape[-2:], 
            hidden_size=best_params['hidden_size'], 
            num_layers=best_params['num_layers'], 
            hidden_size2=best_params['hidden_size2']
        ).to(device)
        model.dropout = nn.Dropout(best_params['dropout_rate'])
    else:  # cnn_attention_lstm
        model = CNNAttentionLSTMModel(
            input_shape=X_train.shape[-2:], 
            num_heads=best_params['num_heads'], 
            out_channels=best_params['out_channels'], 
            kernel_size=best_params['kernel_size'], 
            hidden_size=best_params['hidden_size'], 
            hidden_size2=best_params['hidden_size2']
        ).to(device)
    
    # 训练
    history, _ = train_model(
        model, X_train_torch, y_train_torch, X_val_torch, y_val_torch, 
        epochs=50, batch_size=batch_size, learning_rate=learning_rate, device=device
    )
    
    # 评估
    model_name = 'LSTM_best' if model_type == 'lstm' else 'CNN-Attention-LSTM_best'
    y_pred_actual = evaluate_model(model, X_test_torch, y_test, y_test_inv, scaler_y, model_name)
    #plot_results(y_pred_actual, y_test_inv, history, model_name)
    
    # 保存模型
    torch.save(model.state_dict(), f'./solar/model/{model_name}_best.pth')
    
    return model, history, y_pred_actual

# Load and preprocess data
df_solar = pd.read_csv("./solar/datasets_fill_NAN/fill_pro_Trina_23.4_87-Site_DKA-M9_A+C-Phases.csv")
#利用前100000的数据进行超参数的筛选
df_solar = df_solar.iloc[:100000]
print(df_solar.head())
#检查是否存在空值列，如果有则抛弃
print(df_solar.isnull().sum())
df_solar.drop(["Wind_Speed"], axis=1, inplace=True)
df_solar['timestamp'] = pd.to_datetime(df_solar['timestamp'])
df_solar = df_solar.set_index('timestamp')
df_solar.drop(["Active_Energy_Delivered_Received"], axis=1, inplace=True)
df_solar.drop(["Current_Phase_Average"], axis=1, inplace=True)
X = df_solar.drop(['Active_Power'], axis=1)
y = df_solar[['Active_Power']]
print(X.head())
# Apply PCA
params_pca = {'cum_variance': 0.8, 'if_apply': True}
X_pca = apply_PCA(X, **params_pca)
print(f"PCA output shape: {X_pca.shape}")

# Split data
train_cutoff = int(0.8 * X_pca.shape[0])
val_cutoff = int(0.9 * X_pca.shape[0])

scaler_y = MinMaxScaler()
scaler_y.fit(y[:train_cutoff])
y_norm = scaler_y.transform(y)

hist_size = 24
data_norm = np.concatenate((X_pca, y_norm), axis=1)

X_train, y_train = windowing(data_norm[:train_cutoff, :], data_norm[:train_cutoff, -1], hist_size)
X_val, y_val = windowing(data_norm[train_cutoff:val_cutoff, :], data_norm[train_cutoff:val_cutoff, -1], hist_size)
X_test, y_test = windowing(data_norm[val_cutoff:, :], data_norm[val_cutoff:, -1], hist_size)
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Convert data to PyTorch tensors
X_train_torch = torch.FloatTensor(X_train).to(device)
y_train_torch = torch.FloatTensor(y_train).to(device)
X_val_torch = torch.FloatTensor(X_val).to(device)
y_val_torch = torch.FloatTensor(y_val).to(device)
X_test_torch = torch.FloatTensor(X_test).to(device)

# 优化 LSTMModel
print("Optimizing LSTMModel...")
best_params_lstm, best_value_lstm = optimize_hyperparameters('lstm', n_trials=50)
model_lstm, history_lstm, y_pred_actual_lstm = train_and_evaluate_best_model('lstm', best_params_lstm)

# # 优化 CNNAttentionLSTMModel
# print("Optimizing CNNAttentionLSTMModel...")
# best_params_cnn, best_value_cnn = optimize_hyperparameters('cnn_attention_lstm', n_trials=5)
# model_cnn, history_cnn, y_pred_actual_cnn = train_and_evaluate_best_model('cnn_attention_lstm', best_params_cnn)

# # Inputs
# epochs = 50
# batch_size = 64

# # Train and evaluate LSTM
# model_lstm = LSTMModel(X_train.shape[-2:]).to(device)
# history_lstm = train_model(model_lstm, X_train_torch, y_train_torch, X_val_torch, y_val_torch, epochs, batch_size)
# y_pred_actual_lstm = evaluate_model(model_lstm, X_test_torch, y_test, y_test_inv, scaler_y, 'LSTM')
# plot_results(y_pred_actual_lstm, y_test_inv, history_lstm, 'LSTM')

# model_cnn_attention_lstm = CNNAttentionLSTMModel(X_train.shape[-2:], num_heads = 4).to(device)
# history_cnn_attention_lstm = train_model(model_cnn_attention_lstm, X_train_torch, y_train_torch, X_val_torch, y_val_torch, epochs, batch_size)
# y_pred_actual_cnn_attention_lstm = evaluate_model(model_cnn_attention_lstm, X_test_torch, y_test, y_test_inv, scaler_y, 'CNN-Attention-LSTM')
# plot_results(y_pred_actual_cnn_attention_lstm, y_test_inv, history_cnn_attention_lstm, 'CNN-Attention-LSTM')
