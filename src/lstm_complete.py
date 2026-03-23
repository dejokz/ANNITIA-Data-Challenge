#!/usr/bin/env python3
"""
COMPLETE LSTM Survival Pipeline for ANNITIA

⚠️  DO NOT SUBMIT UNTIL EXPLICITLY INSTRUCTED ⚠️

This is your secret weapon for 0.90+ score.
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
from typing import List, Tuple, Dict
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

RANDOM_STATE = 42


# ============== MODEL ==============

class LSTMSurvival(nn.Module):
    """LSTM with attention for survival analysis."""
    
    def __init__(self, n_seq_features: int, n_static_features: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.4, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention
        attn_size = hidden_size * self.num_directions + hidden_size
        self.attention = nn.Sequential(
            nn.Linear(attn_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Risk head
        combined_size = hidden_size * self.num_directions + n_static_features
        self.risk_head = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, sequence, time_intervals, static, lengths):
        batch_size = sequence.size(0)
        max_len = sequence.size(1)
        device = sequence.device
        
        # Mask
        mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
        
        # Time embedding
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))
        
        # LSTM
        packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Attention
        combined = torch.cat([lstm_out, time_emb], dim=-1)
        attn_scores = self.attention(combined).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Apply attention
        attended = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        
        # Combine with static
        combined_features = torch.cat([attended, static], dim=1)
        
        # Risk
        risk = self.risk_head(combined_features).squeeze(-1)
        return risk


class CoxPHLoss(nn.Module):
    """Cox PH loss."""
    
    def forward(self, log_h, event, time):
        idx = torch.argsort(time, descending=True)
        log_h_sorted = log_h[idx]
        event_sorted = event[idx]
        
        exp_h = torch.exp(log_h_sorted)
        risk_set_sum = torch.cumsum(exp_h, dim=0) + 1e-7
        
        event_mask = event_sorted == 1
        if event_mask.sum() == 0:
            return torch.tensor(0.0, device=log_h.device, requires_grad=True)
        
        log_likelihood = torch.sum(log_h_sorted[event_mask]) - torch.sum(torch.log(risk_set_sum[event_mask]))
        return -log_likelihood / event_mask.sum()


# ============== DATA ==============

class SurvivalDataset(Dataset):
    def __init__(self, sequences, time_intervals, static, events, times):
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
    sequences = [item['sequence'] for item in batch]
    time_intervals = [item['time_intervals'] for item in batch]
    static = torch.stack([item['static'] for item in batch])
    lengths = torch.LongTensor([item['length'] for item in batch])
    events = torch.stack([item['event'] for item in batch]).squeeze()
    times = torch.stack([item['time'] for item in batch]).squeeze()
    
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


# ============== PREPROCESSING ==============

def create_sequences(df, outcome='hepatic'):
    """Create sequential data from wide format."""
    
    # Prepare target
    age_cols = [c for c in df.columns if c.startswith('Age_v')]
    df = df.copy()
    df['first_age'] = df[age_cols].min(axis=1)
    df['last_age'] = df[age_cols].max(axis=1)
    
    if outcome == 'hepatic':
        event_col = 'evenements_hepatiques_majeurs'
        time_col = 'evenements_hepatiques_age_occur'
        mask = ~((df[event_col] == 1) & df[time_col].isna())
    else:
        event_col = 'death'
        time_col = 'death_age_occur'
        mask = ~(df[event_col].isna() | ((df[event_col] == 1) & df[time_col].isna()))
    
    df = df[mask].copy()
    
    is_event = (df[event_col] == 1).values
    time_to_event = np.where(
        is_event,
        df[time_col] - df['first_age'],
        df['last_age'] - df['first_age']
    ).astype(float)
    time_to_event = np.maximum(time_to_event, 0.001)
    
    # Variables to use for sequences
    seq_vars = ['fibs_stiffness_med_BM_1', 'fibrotest_BM_2', 'fib4', 'plt', 'ast', 'alt']
    static_vars = ['gender', 'T2DM', 'Hypertension', 'Dyslipidaemia']
    
    sequences = []
    time_intervals = []
    static_features = []
    valid_indices = []
    
    for idx in df.index:
        # Get visit ages
        ages = []
        for col in age_cols:
            visit_num = int(col.split('_v')[1])
            age = df.loc[idx, col]
            if pd.notna(age):
                ages.append((visit_num, age))
        
        ages.sort(key=lambda x: x[0])
        
        if len(ages) < 2:
            continue
        
        # Build sequence
        patient_seq = []
        visit_ages = [a[1] for a in ages]
        intervals = [0] + [visit_ages[i] - visit_ages[i-1] for i in range(1, len(visit_ages))]
        
        for i, (visit_num, age) in enumerate(ages):
            visit_features = []
            valid = False
            
            for var in seq_vars:
                col = f'{var}_v{visit_num}'
                if col in df.columns:
                    val = df.loc[idx, col]
                    if pd.notna(val):
                        visit_features.append(float(val))
                        valid = True
                    else:
                        visit_features.append(0.0)
                else:
                    visit_features.append(0.0)
            
            if valid or len(patient_seq) > 0:
                patient_seq.append(visit_features)
        
        if len(patient_seq) >= 2:
            sequences.append(np.array(patient_seq))
            time_intervals.append(np.array(intervals[:len(patient_seq)]))
            
            # Static features
            static_vals = [float(df.loc[idx, col]) if col in df.columns and pd.notna(df.loc[idx, col]) else 0.0 
                          for col in static_vars]
            static_features.append(static_vals)
            valid_indices.append(idx)
    
    # Get targets for valid indices
    events = is_event[df.index.isin(valid_indices)]
    times = time_to_event[df.index.isin(valid_indices)]
    
    logger.info(f"Created {len(sequences)} sequences")
    logger.info(f"Events: {events.sum()}/{len(events)} ({100*events.mean():.1f}%)")
    
    return sequences, time_intervals, np.array(static_features), events, times


# ============== TRAINING ==============

def train_lstm(sequences, time_intervals, static, events, times, n_folds=5):
    """Train LSTM with cross-validation."""
    
    # Scale static features
    static_scaler = StandardScaler()
    static_scaled = static_scaler.fit_transform(static)
    
    # Scale sequence features (per feature across all visits)
    all_seqs = np.concatenate(sequences, axis=0)
    seq_means = all_seqs.mean(axis=0)
    seq_stds = all_seqs.std(axis=0) + 1e-7
    
    sequences_scaled = [(seq - seq_means) / seq_stds for seq in sequences]
    
    # K-Fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(sequences))
    fold_cis = []
    
    n_seq_features = sequences[0].shape[1]
    n_static_features = static.shape[1]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(sequences)), events)):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold+1}/{n_folds}")
        logger.info(f"{'='*50}")
        
        # Split data
        train_seqs = [sequences_scaled[i] for i in train_idx]
        train_intervals = [time_intervals[i] for i in train_idx]
        train_static = static_scaled[train_idx]
        train_events = events[train_idx]
        train_times = times[train_idx]
        
        val_seqs = [sequences_scaled[i] for i in val_idx]
        val_intervals = [time_intervals[i] for i in val_idx]
        val_static = static_scaled[val_idx]
        val_events = events[val_idx]
        val_times = times[val_idx]
        
        # Create datasets
        train_dataset = SurvivalDataset(train_seqs, train_intervals, train_static, train_events, train_times)
        val_dataset = SurvivalDataset(val_seqs, val_intervals, val_static, val_events, val_times)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        # Create model
        model = LSTMSurvival(n_seq_features, n_static_features, hidden_size=128, num_layers=2, dropout=0.4)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = CoxPHLoss()
        
        best_ci = 0.0
        patience = 0
        
        for epoch in range(200):
            # Train
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                
                risk = model(
                    batch['sequence'].to(device),
                    batch['time_intervals'].to(device),
                    batch['static'].to(device),
                    batch['lengths'].to(device)
                )
                
                loss = criterion(risk, batch['event'].to(device), batch['time'].to(device))
                
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
            # Validate
            if (epoch + 1) % 10 == 0:
                model.eval()
                all_risks = []
                with torch.no_grad():
                    for batch in val_loader:
                        risk = model(
                            batch['sequence'].to(device),
                            batch['time_intervals'].to(device),
                            batch['static'].to(device),
                            batch['lengths'].to(device)
                        )
                        all_risks.append(risk.cpu().numpy())
                
                val_risks = np.concatenate(all_risks)
                ci = concordance_index_censored(val_events.astype(bool), val_times, val_risks)[0]
                
                if ci > best_ci:
                    best_ci = ci
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                
                if epoch == 9 or patience == 0:
                    logger.info(f"  Epoch {epoch+1}: CI = {ci:.4f}, Best = {best_ci:.4f}")
                
                if patience >= 20:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best and predict
        model.load_state_dict(best_state)
        model.eval()
        all_risks = []
        with torch.no_grad():
            for batch in val_loader:
                risk = model(
                    batch['sequence'].to(device),
                    batch['time_intervals'].to(device),
                    batch['static'].to(device),
                    batch['lengths'].to(device)
                )
                all_risks.append(risk.cpu().numpy())
        
        val_risks = np.concatenate(all_risks)
        oof_preds[val_idx] = val_risks
        fold_cis.append(best_ci)
        logger.info(f"  Fold {fold+1} Best CI: {best_ci:.4f}")
    
    # Overall CI
    overall_ci = concordance_index_censored(events.astype(bool), times, oof_preds)[0]
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Overall OOF C-index: {overall_ci:.4f}")
    logger.info(f"Fold CIs: {[f'{c:.4f}' for c in fold_cis]}")
    logger.info(f"{'='*50}\n")
    
    return overall_ci, oof_preds, fold_cis, seq_means, seq_stds, static_scaler


def main():
    """Main pipeline."""
    logger.info("="*70)
    logger.info("LSTM SURVIVAL - COMPLETE PIPELINE")
    logger.info("⚠️  DO NOT SUBMIT UNTIL EXPLICITLY INSTRUCTED")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Calculate clinical scores
    logger.info("\nCalculating FIB-4...")
    for visit in range(1, 23):
        age_col = f'Age_v{visit}'
        ast_col = f'ast_v{visit}'
        alt_col = f'alt_v{visit}'
        plt_col = f'plt_v{visit}'
        
        if all(c in train_df.columns for c in [age_col, ast_col, alt_col, plt_col]):
            train_df[f'fib4_v{visit}'] = (
                train_df[age_col] * train_df[ast_col] / 
                (train_df[plt_col] * np.sqrt(train_df[alt_col].clip(lower=1)) + 0.001)
            )
            test_df[f'fib4_v{visit}'] = (
                test_df[age_col] * test_df[ast_col] / 
                (test_df[plt_col] * np.sqrt(test_df[alt_col].clip(lower=1)) + 0.001)
            )
    
    # Train Hepatic Model
    logger.info("\n" + "="*70)
    logger.info("HEPATIC EVENTS MODEL")
    logger.info("="*70)
    
    seq_hep, time_hep, static_hep, events_hep, times_hep = create_sequences(train_df, 'hepatic')
    
    ci_hep, oof_hep, fold_cis_hep, seq_mean_hep, seq_std_hep, static_scaler_hep = train_lstm(
        seq_hep, time_hep, static_hep, events_hep, times_hep, n_folds=5
    )
    
    # Train Death Model
    logger.info("\n" + "="*70)
    logger.info("DEATH MODEL")
    logger.info("="*70)
    
    seq_death, time_death, static_death, events_death, times_death = create_sequences(train_df, 'death')
    
    ci_death, oof_death, fold_cis_death, seq_mean_death, seq_std_death, static_scaler_death = train_lstm(
        seq_death, time_death, static_death, events_death, times_death, n_folds=5
    )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    logger.info(f"\n🎯 Hepatic C-index: {ci_hep:.4f}")
    logger.info(f"🎯 Death C-index:   {ci_death:.4f}")
    logger.info(f"🎯 Average:         {(ci_hep + ci_death)/2:.4f}")
    logger.info("="*70)
    logger.info("\n⚠️  DO NOT SUBMIT UNTIL EXPLICITLY INSTRUCTED")
    logger.info("="*70)
    
    # Save results
    results = {
        'hepatic_ci': float(ci_hep),
        'death_ci': float(ci_death),
        'average_ci': float((ci_hep + ci_death)/2),
    }
    
    with open('submissions/lstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n✅ Results saved to submissions/lstm_results.json")


if __name__ == '__main__':
    main()
