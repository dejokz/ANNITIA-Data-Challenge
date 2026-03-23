#!/usr/bin/env python3
"""
Deep Survival Learning for ANNITIA Challenge - GPU Accelerated

This module implements:
1. LSTM/GRU for sequential NIT trajectories
2. Transformer-based survival model
3. DeepSurv (Cox PH with neural networks)
4. Multi-task learning (hepatic + death)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import logging
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

RANDOM_STATE = 42
N_FOLDS = 5


class LongitudinalDataset(Dataset):
    """
    PyTorch Dataset for longitudinal survival data.
    Returns sequences of measurements over time.
    """
    
    def __init__(self, sequences: List[np.ndarray], static: np.ndarray,
                 events: np.ndarray, times: np.ndarray):
        """
        Args:
            sequences: List of (n_visits, n_features) arrays
            static: Static features (n_patients, n_static)
            events: Event indicators (n_patients,)
            times: Survival times (n_patients,)
        """
        self.sequences = sequences
        self.static = static
        self.events = events
        self.times = times
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'static': torch.FloatTensor(self.static[idx]),
            'length': len(self.sequences[idx]),
            'event': torch.FloatTensor([self.events[idx]]),
            'time': torch.FloatTensor([self.times[idx]])
        }


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    sequences = [item['sequence'] for item in batch]
    static = torch.stack([item['static'] for item in batch])
    lengths = torch.LongTensor([item['length'] for item in batch])
    events = torch.stack([item['event'] for item in batch]).squeeze()
    times = torch.stack([item['time'] for item in batch]).squeeze()
    
    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True)
    
    return {
        'sequence': sequences_padded,
        'static': static,
        'lengths': lengths,
        'event': events,
        'time': times
    }


class LSTMSurvival(nn.Module):
    """
    LSTM-based survival model.
    
    Processes longitudinal sequences and outputs risk scores.
    Uses attention mechanism to weigh important visits.
    """
    
    def __init__(self, n_seq_features: int, n_static_features: int,
                 hidden_size: int = 128, n_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        
        # LSTM for sequential data
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.n_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Combine LSTM output with static features
        combined_size = hidden_size * self.n_directions + n_static_features
        
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, sequence, static, lengths):
        """
        Forward pass.
        
        Args:
            sequence: (batch, max_seq_len, n_features)
            static: (batch, n_static_features)
            lengths: (batch,) actual lengths
        
        Returns:
            risk_scores: (batch,) higher = higher risk
        """
        batch_size = sequence.size(0)
        
        # Pack sequence for efficient LSTM processing
        packed = pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Attention over visits
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        
        # Create mask for valid visits
        mask = torch.arange(lstm_out.size(1), device=lstm_out.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        
        attn_weights = attn_weights.masked_fill(~mask.bool(), float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden * dirs)
        
        # Combine with static features
        combined = torch.cat([attended, static], dim=1)
        
        # Predict risk
        risk = self.risk_head(combined).squeeze(-1)
        
        return risk


class TransformerSurvival(nn.Module):
    """
    Transformer-based survival model.
    
    Uses self-attention to capture long-range dependencies in longitudinal data.
    Better for capturing patterns across many visits.
    """
    
    def __init__(self, n_seq_features: int, n_static_features: int,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(n_seq_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Combine with static
        combined_size = d_model + n_static_features
        
        # Risk head
        self.risk_head = nn.Sequential(
            nn.Linear(combined_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, sequence, static, lengths):
        """Forward pass with masking."""
        batch_size, max_len, _ = sequence.shape
        
        # Create padding mask
        padding_mask = torch.arange(max_len, device=sequence.device)[None, :] >= lengths[:, None]
        
        # Project and add positional encoding
        x = self.input_proj(sequence)
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Global average pooling (considering actual lengths)
        mask = (~padding_mask).float().unsqueeze(-1)
        x = torch.sum(x * mask, dim=1) / lengths.unsqueeze(1).float()
        
        # Combine with static
        combined = torch.cat([x, static], dim=1)
        
        # Predict
        risk = self.risk_head(combined).squeeze(-1)
        
        return risk


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DeepSurv(nn.Module):
    """
    DeepSurv: Cox PH with neural network.
    
    Treats static features with fully connected layers.
    Good baseline for comparison.
    """
    
    def __init__(self, n_features: int, hidden_sizes: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_size = n_features
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Returns log hazard ratio."""
        return self.network(x).squeeze(-1)


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards negative log-likelihood.
    
    This is the proper loss for survival analysis.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, log_h: torch.Tensor, event: torch.Tensor, time: torch.Tensor):
        """
        Args:
            log_h: Log hazard ratios (batch,)
            event: Event indicators (batch,) 1 if event occurred
            time: Survival times (batch,)
        
        Returns:
            Negative log partial likelihood
        """
        # Sort by time (descending for Cox)
        idx = torch.argsort(time, descending=True)
        log_h = log_h[idx]
        event = event[idx]
        time = time[idx]
        
        # Calculate log partial likelihood
        # log PL = sum_{i: event_i=1} (log_h_i - log(sum_{j: time_j >= time_i} exp(log_h_j)))
        
        exp_h = torch.exp(log_h)
        
        # Cumulative sum of exp(log_h) for risk sets
        risk_set_sum = torch.cumsum(exp_h, dim=0)
        
        # Only consider events
        event_mask = event == 1
        
        if event_mask.sum() == 0:
            return torch.tensor(0.0, device=log_h.device)
        
        log_likelihood = torch.sum(log_h[event_mask]) - torch.sum(torch.log(risk_set_sum[event_mask]))
        
        return -log_likelihood / event_mask.sum()


class DeepSurvivalTrainer:
    """Trainer for deep survival models."""
    
    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = CoxPHLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Forward
            risk = self.model(
                batch['sequence'].to(self.device),
                batch['static'].to(self.device),
                batch['lengths'].to(self.device)
            )
            
            # Loss
            loss = self.criterion(
                risk,
                batch['event'].to(self.device),
                batch['time'].to(self.device)
            )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, np.ndarray]:
        """Evaluate model."""
        self.model.eval()
        all_risks = []
        all_events = []
        all_times = []
        
        with torch.no_grad():
            for batch in dataloader:
                risk = self.model(
                    batch['sequence'].to(self.device),
                    batch['static'].to(self.device),
                    batch['lengths'].to(self.device)
                )
                
                all_risks.append(risk.cpu().numpy())
                all_events.append(batch['event'].numpy())
                all_times.append(batch['time'].numpy())
        
        risks = np.concatenate(all_risks)
        events = np.concatenate(all_events)
        times = np.concatenate(all_times)
        
        # Calculate C-index
        try:
            ci = concordance_index_censored(
                events.astype(bool), times, risks
            )[0]
        except:
            ci = 0.5
        
        return ci, risks
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            n_epochs: int = 100, patience: int = 10) -> Dict:
        """Train with early stopping."""
        best_ci = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_ci': []}
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_ci, _ = self.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_ci'].append(val_ci)
            
            self.scheduler.step(train_loss)
            
            if val_ci > best_ci:
                best_ci = val_ci
                patience_counter = 0
                # Save best model
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f}, Val CI: {val_ci:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_state)
        
        return history


def prepare_sequence_data(df: pd.DataFrame, feature_cols: List[str],
                          static_cols: List[str]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Convert wide-format dataframe to sequences.
    
    Returns:
        sequences: List of (n_visits, n_features) per patient
        static: (n_patients, n_static) static features
    """
    sequences = []
    static_features = []
    
    # Group features by variable
    var_groups = {}
    for col in feature_cols:
        if '_v' in col:
            parts = col.rsplit('_v', 1)
            if len(parts) == 2 and parts[1].isdigit():
                var_name = parts[0]
                visit_num = int(parts[1])
                if var_name not in var_groups:
                    var_groups[var_name] = []
                var_groups[var_name].append((visit_num, col))
    
    # Sort each group by visit number
    for var_name in var_groups:
        var_groups[var_name].sort(key=lambda x: x[0])
    
    # Get ordered variable names
    ordered_vars = sorted(var_groups.keys())
    
    for idx in df.index:
        patient_data = []
        
        # For each visit, collect all variables
        max_visits = max([len(v) for v in var_groups.values()])
        
        for visit_idx in range(max_visits):
            visit_features = []
            valid = False
            
            for var_name in ordered_vars:
                if visit_idx < len(var_groups[var_name]):
                    _, col_name = var_groups[var_name][visit_idx]
                    val = df.loc[idx, col_name]
                    if pd.notna(val):
                        valid = True
                    visit_features.append(val if pd.notna(val) else 0.0)
                else:
                    visit_features.append(0.0)
            
            if valid:
                patient_data.append(visit_features)
        
        # Convert to array, drop trailing NaN visits
        if patient_data:
            seq_array = np.array(patient_data)
            # Remove visits that are all zeros/NaN
            valid_visits = ~np.all(seq_array == 0, axis=1)
            seq_array = seq_array[valid_visits]
            sequences.append(seq_array)
        else:
            sequences.append(np.zeros((1, len(ordered_vars))))
        
        # Static features
        static_vals = []
        for col in static_cols:
            if col in df.columns:
                val = df.loc[idx, col]
                static_vals.append(val if pd.notna(val) else 0.0)
            else:
                static_vals.append(0.0)
        static_features.append(static_vals)
    
    return sequences, np.array(static_features)


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("DEEP SURVIVAL LEARNING - GPU ACCELERATED")
    logger.info("="*70)
    
    # Load data
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    logger.info(f"Data loaded: train={train_df.shape}, test={test_df.shape}")
    
    # For now, use static features only (DeepSurv)
    # TODO: Add sequence preparation for LSTM/Transformer
    
    # Feature engineering (reuse from previous)
    from pipeline import TrajectoryFeatureEngineer
    
    engineer = TrajectoryFeatureEngineer()
    X_train = engineer.transform(train_df)
    X_test = engineer.transform(test_df)
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    logger.info(f"Features: {len(common_cols)}")
    
    # Prepare targets
    def prepare_survival_target(df, outcome='hepatic'):
        df = df.copy()
        age_cols = [c for c in df.columns if c.startswith('Age_v')]
        df['last_observed_age'] = df[age_cols].max(axis=1)
        df['first_visit_age'] = df[age_cols].min(axis=1)
        
        if outcome == 'hepatic':
            event_col = 'evenements_hepatiques_majeurs'
            age_occur_col = 'evenements_hepatiques_age_occur'
            is_event = df[event_col] == 1
            invalid = is_event & df[age_occur_col].isna()
            df_valid = df[~invalid].copy()
        else:
            event_col = 'death'
            age_occur_col = 'death_age_occur'
            is_event = df[event_col] == 1
            unknown = df[event_col].isna()
            invalid = is_event & df[age_occur_col].isna()
            df_valid = df[~(unknown | invalid)].copy()
        
        is_event_v = (df_valid[event_col] == 1)
        time_values = np.where(
            is_event_v,
            df_valid[age_occur_col] - df_valid['first_visit_age'],
            df_valid['last_observed_age'] - df_valid['first_visit_age']
        ).astype(float)
        time_values = np.maximum(time_values, 0.001)
        
        return df_valid, is_event_v.values.astype(np.float32), time_values.astype(np.float32)
    
    # Train DeepSurv model
    logger.info("\n" + "="*70)
    logger.info("TRAINING DEEPSURV MODEL")
    logger.info("="*70)
    
    df_hep, events_hep, times_hep = prepare_survival_target(train_df, 'hepatic')
    X_hep = X_train.loc[df_hep.index].fillna(X_train.median())
    
    # Scale features
    scaler = StandardScaler()
    X_hep_scaled = scaler.fit_transform(X_hep)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_hep_scaled)
    events_tensor = torch.FloatTensor(events_hep)
    times_tensor = torch.FloatTensor(times_hep)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, events_tensor, times_tensor)
    
    # Cross-validation
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X_hep))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_hep_scaled, events_hep)):
        logger.info(f"\nFold {fold+1}/5")
        
        # Create model
        model = DeepSurv(n_features=X_hep_scaled.shape[1], hidden_sizes=[256, 128, 64], dropout=0.4)
        model = model.to(device)
        
        # Prepare data
        X_train_fold = X_tensor[train_idx].to(device)
        events_train = events_tensor[train_idx].to(device)
        times_train = times_tensor[train_idx].to(device)
        
        X_val_fold = X_tensor[val_idx].to(device)
        events_val = events_tensor[val_idx]
        times_val = times_tensor[val_idx]
        
        # Train
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = CoxPHLoss()
        
        best_ci = 0.0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            
            log_h = model(X_train_fold)
            loss = criterion(log_h, events_train, times_train)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_risk = model(X_val_fold).cpu().numpy()
                    ci = concordance_index_censored(
                        events_val.numpy().astype(bool),
                        times_val.numpy(),
                        val_risk
                    )[0]
                    if ci > best_ci:
                        best_ci = ci
                logger.info(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Val CI={ci:.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_risk = model(X_val_fold).cpu().numpy()
            oof_preds[val_idx] = val_risk
        
        logger.info(f"  Best CI: {best_ci:.4f}")
    
    # Overall OOF CI
    overall_ci = concordance_index_censored(
        events_hep.astype(bool), times_hep, oof_preds
    )[0]
    
    logger.info(f"\nOverall OOF C-index: {overall_ci:.4f}")
    
    # Save results
    results = {
        'deep_surv_ci': float(overall_ci),
        'n_features': X_hep_scaled.shape[1]
    }
    
    with open('submissions/deep_surv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✅ Results saved!")


if __name__ == '__main__':
    main()
