#!/usr/bin/env python3
"""
LSTM-based Survival Analysis for ANNITIA Challenge

This implements:
1. Sequential data preparation (visit-level features over time)
2. LSTM with attention mechanism
3. Cox PH loss for proper survival modeling
4. Time-aware embeddings (irregular visit intervals)
5. GPU acceleration

DO NOT SUBMIT until explicitly instructed.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

RANDOM_STATE = 42
N_FOLDS = 5


@dataclass
class Config:
    """Model configuration."""
    # LSTM
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.4  # Heavy dropout for regularization
    bidirectional: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 200
    patience: int = 20
    
    # Features
    max_seq_len: int = 22  # Max visits
    n_static_features: int = 5  # gender, T2DM, etc.


class SequentialSurvivalDataset(Dataset):
    """
    Dataset for sequential survival data.
    
    Each sample is a patient's longitudinal trajectory:
    - sequences: (n_visits, n_features) array of NIT measurements
    - time_intervals: (n_visits,) time since previous visit (for irregular sampling)
    - static: Static features (demographics)
    - event: 1 if event occurred, 0 if censored
    - time_to_event: Survival time
    """
    
    def __init__(self, sequences: List[np.ndarray], 
                 time_intervals: List[np.ndarray],
                 static: np.ndarray,
                 events: np.ndarray, 
                 times: np.ndarray):
        self.sequences = sequences
        self.time_intervals = time_intervals
        self.static = static
        self.events = events
        self.times = times
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'time_intervals': torch.FloatTensor(self.time_intervals[idx]),
            'static': torch.FloatTensor(self.static[idx]),
            'length': len(self.sequences[idx]),
            'event': torch.FloatTensor([self.events[idx]]),
            'time': torch.FloatTensor([self.times[idx]])
        }


def collate_fn(batch):
    """Custom collate for variable-length sequences."""
    sequences = [item['sequence'] for item in batch]
    time_intervals = [item['time_intervals'] for item in batch]
    static = torch.stack([item['static'] for item in batch])
    lengths = torch.LongTensor([item['length'] for item in batch])
    events = torch.stack([item['event'] for item in batch]).squeeze()
    times = torch.stack([item['time'] for item in batch]).squeeze()
    
    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    time_intervals_padded = pad_sequence(time_intervals, batch_first=True, padding_value=0.0)
    
    return {
        'sequence': sequences_padded,
        'time_intervals': time_intervals_padded,
        'static': static,
        'lengths': lengths,
        'event': events,
        'time': times
    }


class TimeAwareLSTM(nn.Module):
    """
    LSTM with time-aware attention for survival analysis.
    
    Key innovations:
    1. Time interval embeddings (handles irregular visit spacing)
    2. Attention mechanism over visits (learns which visits matter)
    3. Combines sequential + static features
    4. Outputs risk scores (higher = more likely to have event)
    """
    
    def __init__(self, n_seq_features: int, n_static_features: int, config: Config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_directions = 2 if config.bidirectional else 1
        
        # Time interval embedding (learns irregular sampling patterns)
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )
        
        # LSTM for sequential data
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Attention mechanism
        # Combines LSTM output + time embedding
        attn_input_size = config.hidden_size * self.num_directions + config.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(attn_input_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )
        
        # Combine attended sequence + static features
        combined_size = config.hidden_size * self.num_directions + n_static_features
        
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
    def forward(self, sequence, time_intervals, static, lengths):
        """
        Forward pass.
        
        Args:
            sequence: (batch, max_seq_len, n_seq_features)
            time_intervals: (batch, max_seq_len) - time since prev visit
            static: (batch, n_static_features)
            lengths: (batch,) - actual sequence lengths
        
        Returns:
            risk_scores: (batch,) - higher = higher risk
        """
        batch_size = sequence.size(0)
        max_len = sequence.size(1)
        
        # Create padding mask
        device = sequence.device
        mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]  # (batch, max_len)
        
        # Embed time intervals
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))  # (batch, max_len, hidden)
        
        # Pack sequence for efficient LSTM
        packed_seq = pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(packed_seq)  # lstm_out: packed
        
        # Unpack
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # (batch, max_len, hidden * dirs)
        
        # Combine LSTM output with time embedding
        combined = torch.cat([lstm_out, time_emb], dim=-1)  # (batch, max_len, hidden * dirs + hidden)
        
        # Attention over visits
        attn_scores = self.attention(combined).squeeze(-1)  # (batch, max_len)
        
        # Mask padding
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, max_len)
        
        # Apply attention (weighted sum of visit representations)
        attended = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden * dirs)
        
        # Combine with static features
        combined_features = torch.cat([attended, static], dim=1)
        
        # Predict risk
        risk = self.risk_head(combined_features).squeeze(-1)  # (batch,)
        
        return risk


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards negative log-likelihood.
    
    Proper loss function for survival analysis.
    """
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(self, log_h: torch.Tensor, event: torch.Tensor, time: torch.Tensor):
        """
        Args:
            log_h: Log hazard ratios (batch,) - output from model
            event: Event indicators (batch,) - 1 if event occurred
            time: Survival times (batch,)
        
        Returns:
            Negative log partial likelihood (scaled by number of events)
        """
        # Sort by time (descending - standard for Cox)
        idx = torch.argsort(time, descending=True)
        log_h_sorted = log_h[idx]
        event_sorted = event[idx]
        
        # Calculate log partial likelihood
        # For each event patient: log(h_i) - log(sum_{j: time_j >= time_i} h_j)
        
        exp_h = torch.exp(log_h_sorted)
        
        # Cumulative sum for risk sets
        risk_set_sum = torch.cumsum(exp_h, dim=0) + self.eps
        
        # Only consider events
        event_mask = event_sorted == 1
        
        if event_mask.sum() == 0:
            return torch.tensor(0.0, device=log_h.device, requires_grad=True)
        
        # Log partial likelihood
        log_likelihood = torch.sum(log_h_sorted[event_mask]) - torch.sum(torch.log(risk_set_sum[event_mask]))
        
        # Normalize by number of events
        return -log_likelihood / event_mask.sum()


class LSTMSurvivalTrainer:
    """Trainer for LSTM survival model."""
    
    def __init__(self, model: nn.Module, config: Config, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5, verbose=True
        )
        self.criterion = CoxPHLoss()
        self.best_state = None
        
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
                batch['time_intervals'].to(self.device),
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
            if loss.requires_grad and loss.item() != 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model."""
        self.model.eval()
        all_risks = []
        all_events = []
        all_times = []
        
        with torch.no_grad():
            for batch in dataloader:
                risk = self.model(
                    batch['sequence'].to(self.device),
                    batch['time_intervals'].to(self.device),
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
        
        return ci, risks, events, times
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Train with early stopping."""
        best_ci = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_ci': []}
        
        logger.info(f"\nTraining LSTM for max {self.config.n_epochs} epochs...")
        
        for epoch in range(self.config.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_ci, _, _, _ = self.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_ci'].append(val_ci)
            
            self.scheduler.step(val_ci)
            
            if val_ci > best_ci:
                best_ci = val_ci
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{self.config.n_epochs} - Loss: {train_loss:.4f}, Val CI: {val_ci:.4f}, Best: {best_ci:.4f}")
            
            if patience_counter >= self.config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return history, best_ci


def prepare_sequential_data(df: pd.DataFrame, 
                           seq_feature_cols: List[str],
                           static_cols: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert wide-format dataframe to sequential format.
    
    Returns:
        sequences: List of (n_visits, n_seq_features) per patient
        time_intervals: List of (n_visits,) time intervals per patient
        static: (n_patients, n_static) static features
        events: (n_patients,) event indicators
        times: (n_patients,) survival times
    """
    logger.info("Converting to sequential format...")
    
    sequences = []
    time_intervals = []
    static_features = []
    events_list = []
    times_list = []
    
    # Group sequence features by variable
    var_groups = {}
    for col in seq_feature_cols:
        if '_v' in col:
            parts = col.rsplit('_v', 1)
            if len(parts) == 2 and parts[1].isdigit():
                var_name = parts[0]
                visit_num = int(parts[1])
                if var_name not in var_groups:
                    var_groups[var_name] = []
                var_groups[var_name].append((visit_num, col))
    
    # Sort by visit number
    for var_name in var_groups:
        var_groups[var_name].sort(key=lambda x: x[0])
    
    # Get ordered variable names
    ordered_vars = sorted(var_groups.keys())
    logger.info(f"  Variables: {ordered_vars}")
    
    # Process each patient
    for idx in df.index:
        patient_seq = []
        patient_times = []
        
        # Get ages for this patient
        age_cols = [c for c in df.columns if c.startswith('Age_v')]
        ages = []
        for col in age_cols:
            visit_num = int(col.split('_v')[1])
            age = df.loc[idx, col]
            if pd.notna(age):
                ages.append((visit_num, age))
        
        ages.sort(key=lambda x: x[0])
        
        if len(ages) < 2:
            # Skip patients with < 2 visits
            continue
        
        # Calculate time intervals
        visit_ages = [a[1] for a in ages]
        intervals = [0] + [visit_ages[i] - visit_ages[i-1] for i in range(1, len(visit_ages))]
        
        # For each visit, collect all features
        for visit_idx, (visit_num, age) in enumerate(ages):
            visit_features = []
            
            for var_name in ordered_vars:
                # Find feature for this visit
                feat_val = None
                for v_num, col_name in var_groups[var_name]:
                    if v_num == visit_num:
                        feat_val = df.loc[idx, col_name]
                        break
                
                if feat_val is not None and pd.notna(feat_val):
                    visit_features.append(float(feat_val))
                else:
                    visit_features.append(0.0)  # Pad with 0
            
            if len(visit_features) == len(ordered_vars):
                patient_seq.append(visit_features)
                patient_times.append(intervals[visit_idx])
        
        if len(patient_seq) >= 2:  # Need at least 2 visits
            sequences.append(np.array(patient_seq))
            time_intervals.append(np.array(patient_times))
            
            # Static features
            static_vals = []
            for col in static_cols:
                if col in df.columns:
                    val = df.loc[idx, col]
                    static_vals.append(float(val) if pd.notna(val) else 0.0)
                else:
                    static_vals.append(0.0)
            static_features.append(static_vals)
            
            # Event and time
            # These need to be provided separately
            events_list.append(0)  # Placeholder
            times_list.append(0.0)  # Placeholder
    
    logger.info(f"  Created {len(sequences)} sequences")
    logger.info(f"  Avg visits per patient: {np.mean([len(s) for s in sequences]):.1f}")
    
    return (sequences, time_intervals, 
            np.array(static_features), 
            np.array(events_list), 
            np.array(times_list))


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("LSTM SURVIVAL MODEL - GPU ACCELERATED")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # TODO: Implement full pipeline with proper target preparation
    # This is a skeleton for the LSTM approach
    
    logger.info("\n" + "="*70)
    logger.info("LSTM IMPLEMENTATION READY")
    logger.info("="*70)
    logger.info("\n⚠️  DO NOT SUBMIT UNTIL EXPLICITLY INSTRUCTED")
    logger.info("="*70)


if __name__ == '__main__':
    main()
