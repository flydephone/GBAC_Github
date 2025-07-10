import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import utils

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_np(tensor):
    return tensor.clone().data.cpu().numpy()

class MyDataSet(Dataset):
    """
    Custom PyTorch Dataset that standardizes features (E) and targets (Y),
    and pre-computes one-hot encoding for genetic features (G).
    """
    def __init__(self, X, Y, feature_G, feature_E, scaler_E=None, scaler_Y=None):
        self.X = X
        
        # --- Environmental Feature (E) Scaling ---
        if scaler_E is None:
            self.scaler_E = StandardScaler()
            self.scaler_E.fit(feature_E)
        else:
            self.scaler_E = scaler_E
        
        feature_E_scaled = feature_E.copy()
        feature_E_scaled.iloc[:, :] = self.scaler_E.transform(feature_E)
        self.feature_E = feature_E_scaled

        # --- Target (Y) Scaling ---
        if scaler_Y is None:
            self.scaler_Y = StandardScaler()
            self.scaler_Y.fit(Y)
        else:
            self.scaler_Y = scaler_Y
        
        # Apply scaling to the target variable Y
        self.Y = self.scaler_Y.transform(Y)

        # --- Genetic Feature (G) One-Hot Encoding ---
        self.num_classes = 4
        feature_G_shifted = feature_G + 1
        g_tensor_shifted = torch.tensor(feature_G_shifted.values, dtype=torch.long)
        g_one_hot_all = F.one_hot(g_tensor_shifted, num_classes=self.num_classes)
        self.g_feature_map = {
            name: g_one_hot_all[i].to(torch.float32)
            for i, name in enumerate(feature_G.index)
        }

    def __getitem__(self, idx):
        env_name, hybrid_name = self.X[idx]
        e = torch.tensor(self.feature_E.loc[env_name].values, dtype=torch.float32)
        g_final = self.g_feature_map[hybrid_name]
        y = torch.tensor(self.Y[idx], dtype=torch.float32)

        return e, g_final, y

    def __len__(self):
        return len(self.X)

class Mymodel(nn.Module):
    def __init__(self, env_feature_size, genetic_feature_size, env_embedding_dim=32, gen_embedding_dim=10):
        super(Mymodel, self).__init__()
        
        self.positional_snp_weights = nn.Parameter(torch.randn(genetic_feature_size, 4))
        self.genetic_embed_2 = nn.Linear(genetic_feature_size, gen_embedding_dim)
        
        self.env_embed = nn.Linear(env_feature_size, env_embedding_dim)
        
        combined_dim = gen_embedding_dim + env_embedding_dim
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, env_features, genetic_features):
        g1 = torch.sum(genetic_features * self.positional_snp_weights, dim=2) # -> [batch, num_snps]
        g_embedded = F.relu(self.genetic_embed_2(g1)) # -> [batch, 10]
        e_embedded = F.relu(self.env_embed(env_features)) # -> [batch, 32]
        
        # Concatenate embeddings
        combined = torch.cat((g_embedded, e_embedded), dim=1)
        
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # %% Load data
    with open("Formated_Genotypes.pickle", 'rb') as f:
        genotypes = pickle.load(f)
    with open("Formated_Environment.pickle", 'rb') as f:
        environments = pickle.load(f)
    genotypes.index.name = 'Hybrid'
    
    tra_phenotypes = pd.read_csv('Phenotypes.csv')
    tes_phenotypes = pd.read_csv('Hybrids_to_be_predicted.csv')
    
    tra_phenotypes.rename(columns={'Trait_1': 'Yield'}, inplace=True)
    tra_phenotypes.rename(columns={'Trait_2': 'Moisture'}, inplace=True)
    tes_phenotypes.rename(columns={'Trait_1': 'Yield'}, inplace=True)
    tes_phenotypes.rename(columns={'Trait_2': 'Moisture'}, inplace=True)

    # %% Dataset split
    data_tra, data_val = utils.split_by_environment(tra_phenotypes, split_col='Environment', train_ratio=0.7, seed=0)
    data_tes = tes_phenotypes
    
    X_tra, Y_tra = data_tra[['Environment', 'Hybrid']].values, data_tra[['Yield', 'Moisture']].values
    X_val, Y_val = data_val[['Environment', 'Hybrid']].values, data_val[['Yield', 'Moisture']].values
    X_tes, Y_tes = data_tes[['Environment', 'Hybrid']].values, data_tes[['Yield', 'Moisture']].values
    feature_G = genotypes
    feature_E = environments
    
    # %% Dataloader
    batch_size = 128
    tra_set = MyDataSet(X_tra, Y_tra, feature_G, feature_E)
    tra_DataLoader = DataLoader(tra_set, batch_size=batch_size, shuffle=True)
    val_set = MyDataSet(X_val, Y_val, feature_G, feature_E, scaler_E=tra_set.scaler_E, scaler_Y=tra_set.scaler_Y)
    val_DataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # %% ============== GPU/CPU Device Setup ==============
    # 1. Set the device to a GPU if available, otherwise use the CPU
    
    print(f"Using device: {device}")
    
    # %% Model and optimizer define
    EPOCHS = 20
    LEARNING_RATE = 0.001
    e_size = len(feature_E.columns)
    g_size = len(feature_G.columns)
    model = Mymodel(env_feature_size=e_size, genetic_feature_size=g_size, env_embedding_dim=16, gen_embedding_dim=128)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # %% Training
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for env_batch, genetic_batch, target_batch in tra_DataLoader:
            # 3. Move data batches to the selected device
            env_batch = env_batch.to(device)
            genetic_batch = genetic_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            outputs = model(env_batch, genetic_batch)
            mask = ~torch.isnan(target_batch)
            loss = F.mse_loss(outputs[mask], target_batch[mask])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(tra_DataLoader)
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for env_batch, genetic_batch, target_batch in val_DataLoader:
                # 4. Move validation data batches to the selected device
                env_batch = env_batch.to(device)
                genetic_batch = genetic_batch.to(device)
                target_batch = target_batch.to(device)

                outputs = model(env_batch, genetic_batch)
                mask = ~torch.isnan(target_batch)
                loss = F.mse_loss(outputs[mask], target_batch[mask])
                val_loss += loss.item()
        
        validation_loss = val_loss / len(val_DataLoader)
    
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {validation_loss:.4f}")
    
    print("--- Finished Training ---")
    
    # %% Get predictions for the entire validation set
    model.eval()
    all_predictions_scaled = []
    all_targets_scaled = []
    with torch.no_grad():
        for env_batch, genetic_batch, target_batch in val_DataLoader:
            # 5. Move data to device for final prediction run
            env_batch = env_batch.to(device)
            genetic_batch = genetic_batch.to(device)
            target_batch = target_batch.to(device)
            
            outputs = model(env_batch, genetic_batch)
            all_predictions_scaled.append(outputs)
            all_targets_scaled.append(target_batch)

    all_predictions_scaled = torch.cat(all_predictions_scaled, dim=0)
    all_targets_scaled = torch.cat(all_targets_scaled, dim=0)

    # %% Inverse transform to original scale
    # 6. Move tensors to CPU before converting to NumPy
    predictions_original = tra_set.scaler_Y.inverse_transform(all_predictions_scaled.cpu().numpy())
    targets_original = tra_set.scaler_Y.inverse_transform(all_targets_scaled.cpu().numpy())

    # %% Separate traits
    yield_preds = predictions_original[:, 0]
    yield_actual = targets_original[:, 0]
    moisture_preds = predictions_original[:, 1]
    moisture_actual = targets_original[:, 1]
    
    # %% IMPORTANT: Create masks to filter out NaN values before evaluation
    valid_yield_mask = ~np.isnan(yield_actual)
    valid_moisture_mask = ~np.isnan(moisture_actual)

    # %% Calculate R-squared and RMSE on valid (non-NaN) data only
    r2_yield = r2_score(yield_actual[valid_yield_mask], yield_preds[valid_yield_mask])
    rmse_yield = np.sqrt(mean_squared_error(yield_actual[valid_yield_mask], yield_preds[valid_yield_mask]))

    r2_moisture = r2_score(moisture_actual[valid_moisture_mask], moisture_preds[valid_moisture_mask])
    rmse_moisture = np.sqrt(mean_squared_error(moisture_actual[valid_moisture_mask], moisture_preds[valid_moisture_mask]))

    # %% Create the plots using only the valid data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot for Yield
    ax1.scatter(yield_actual[valid_yield_mask], yield_preds[valid_yield_mask], alpha=0.6, edgecolors='k')
    ax1.plot([yield_actual[valid_yield_mask].min(), yield_actual[valid_yield_mask].max()], 
             [yield_actual[valid_yield_mask].min(), yield_actual[valid_yield_mask].max()], 'r--', lw=2, label='1:1 Line')
    ax1.set_xlabel("Actual Yield", fontsize=12)
    ax1.set_ylabel("Predicted Yield", fontsize=12)
    ax1.set_title("Yield: Actual vs. Predicted", fontsize=14, fontweight='bold')
    ax1.text(0.05, 0.95, f'$R^2 = {r2_yield:.3f}$\n$RMSE = {rmse_yield:.3f}$',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    ax1.grid(True)
    ax1.legend()

    # Plot for Moisture
    ax2.scatter(moisture_actual[valid_moisture_mask], moisture_preds[valid_moisture_mask], alpha=0.6, edgecolors='k', color='green')
    ax2.plot([moisture_actual[valid_moisture_mask].min(), moisture_actual[valid_moisture_mask].max()], 
             [moisture_actual[valid_moisture_mask].min(), moisture_actual[valid_moisture_mask].max()], 'r--', lw=2, label='1:1 Line')
    ax2.set_xlabel("Actual Moisture", fontsize=12)
    ax2.set_ylabel("Predicted Moisture", fontsize=12)
    ax2.set_title("Moisture: Actual vs. Predicted", fontsize=14, fontweight='bold')
    ax2.text(0.05, 0.95, f'$R^2 = {r2_moisture:.3f}$\n$RMSE = {rmse_moisture:.3f}$',
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()