"""
Train all baseline models (CNN, CNN+LSTM, CNN+GRU).

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --model cnn
    python scripts/train_baselines.py --model cnn_lstm
    python scripts/train_baselines.py --model cnn_gru
"""

import os
import sys
import argparse
import logging
import json

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.data.normalization import Normalizer
from src.data.loaders import get_dataloaders
from src.models.cnn_baseline import CNNTrajectoryRegressor
from src.models.cnn_lstm import CNNLSTMRegressor
from src.models.cnn_gru import CNNGRURegressor
from src.train.trainer import Trainer
from src.train.losses import TrajectoryMSELoss, CombinedTrajectoryLoss
from src.train.schedulers import get_scheduler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    'cnn': 'configs/baseline_cnn.yaml',
    'cnn_lstm': 'configs/baseline_cnn_lstm.yaml',
    'cnn_gru': 'configs/baseline_cnn_gru.yaml',
}


def build_model(cfg):
    """Build a model from config dict."""
    m = cfg['model']
    model_type = m['type']

    if model_type == 'cnn':
        return CNNTrajectoryRegressor(
            input_steps=m['input_steps'], input_dim=m['input_dim'],
            output_steps=m['output_steps'], output_dim=m['output_dim'],
            conv_channels=m['conv_channels'], kernel_sizes=m['kernel_sizes'],
            fc_dims=m['fc_dims'], activation=m['activation'], dropout=m['dropout'],
        )
    elif model_type == 'cnn_lstm':
        return CNNLSTMRegressor(
            input_steps=m['input_steps'], input_dim=m['input_dim'],
            output_steps=m['output_steps'], output_dim=m['output_dim'],
            conv_channels=m['conv_channels'], kernel_sizes=m['kernel_sizes'],
            lstm_hidden=m['lstm_hidden'], lstm_layers=m['lstm_layers'],
            lstm_dropout=m['lstm_dropout'], fc_dims=m['fc_dims'],
            activation=m['activation'], dropout=m['dropout'],
        )
    elif model_type == 'cnn_gru':
        return CNNGRURegressor(
            input_steps=m['input_steps'], input_dim=m['input_dim'],
            output_steps=m['output_steps'], output_dim=m['output_dim'],
            conv_channels=m['conv_channels'], kernel_sizes=m['kernel_sizes'],
            gru_hidden=m['gru_hidden'], gru_layers=m['gru_layers'],
            gru_dropout=m['gru_dropout'], fc_dims=m['fc_dims'],
            activation=m['activation'], dropout=m['dropout'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_loss(cfg):
    """Build loss function from config dict."""
    loss_cfg = cfg.get('loss', {})
    loss_type = loss_cfg.get('type', 'mse')
    if loss_type == 'combined':
        return CombinedTrajectoryLoss(
            mse_weight=loss_cfg.get('mse_weight', 1.0),
            smoothness_weight=loss_cfg.get('smoothness_weight', 0.01),
            endpoint_weight=loss_cfg.get('endpoint_weight', 1.0),
        )
    else:
        return TrajectoryMSELoss(
            endpoint_weight=loss_cfg.get('endpoint_weight', 1.0),
        )


def train_single_model(model_name, config_path):
    """Train a single baseline model."""
    logger.info(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")

    cfg = load_config(config_path)
    tcfg = cfg['training']
    set_seed(tcfg['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Load normalization stats
    norm_path = os.path.join(cfg['data']['metadata_dir'], 'normalization_stats.json')
    normalizer = Normalizer.load(norm_path)

    # Data loaders
    loaders = get_dataloaders(
        data_dir=cfg['data']['dir'],
        normalizer=normalizer,
        batch_size=tcfg['batch_size'],
        input_mode=cfg['data']['input_mode'],
    )

    # Model
    model = build_model(cfg)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")

    # Optimizer
    if tcfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg['lr'])
    elif tcfg['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=tcfg['lr'], momentum=0.9)

    # Scheduler
    scheduler = get_scheduler(optimizer, tcfg['scheduler'], T_max=tcfg['epochs'])

    # Loss
    loss_fn = build_loss(cfg)

    # Trainer
    ckpt_dir = os.path.join('results', 'checkpoints', model_name)
    log_dir = os.path.join('results', 'logs', model_name)
    trainer = Trainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler,
        device=device, checkpoint_dir=ckpt_dir, log_dir=log_dir,
        use_amp=tcfg.get('use_amp', False),
        grad_clip=tcfg.get('grad_clip', None),
        early_stop_patience=tcfg.get('early_stop_patience', 0),
    )

    history = trainer.fit(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=tcfg['epochs'],
        model_name=model_name,
    )

    # Save training history
    metrics_dir = os.path.join('results', 'metrics', 'baselines')
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, f'{model_name}_history.json'), 'w') as f:
        json.dump(history, f)

    logger.info(f"Best val loss: {trainer.best_val_loss:.6f}")
    return trainer.best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--model', default='all',
                        choices=['all', 'cnn', 'cnn_lstm', 'cnn_gru'],
                        help='Which model to train')
    args = parser.parse_args()

    if args.model == 'all':
        models_to_train = MODEL_CONFIGS
    else:
        models_to_train = {args.model: MODEL_CONFIGS[args.model]}

    results = {}
    for name, config_path in models_to_train.items():
        best_val = train_single_model(name, config_path)
        results[name] = best_val

    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, val_loss in results.items():
        logger.info(f"  {name:12s}: best_val_loss = {val_loss:.6f}")


if __name__ == '__main__':
    main()