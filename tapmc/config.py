from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

# 模型設定檔

config_patchtst = {
    'input_size': tune.choice([5, 7, 14]),
    'max_steps': 500,
    'val_check_steps': tune.choice([20, 30]),
    'early_stop_patience_steps': tune.choice([3]),
    'encoder_layers': tune.choice([4, 8, 12]),
    'dropout': tune.choice([0.2, 0.3]),
    'head_dropout': tune.choice([0.2, 0.3]),
    'patch_len': tune.choice([12, 15]),
    'linear_hidden_size': tune.choice([128, 256]),
    'attn_dropout': tune.choice([0.2, 0.3]),
    'hidden_size': tune.choice([128, 256, 512]),
    'n_heads': tune.choice([8, 16, 32]),
    'activation': 'gelu',
    'learning_rate': tune.loguniform(1e-4, 1e-2),
    'batch_size': tune.choice([2, 4, 8])
}

config_lstm = {
    'input_size': tune.choice([5, 7, 14]),
    'max_steps': 500,
    'val_check_steps': tune.choice([20, 30]),
    'early_stop_patience_steps': tune.choice([3]),
    'encoder_n_layers': tune.choice([3, 5, 7]),
    'encoder_dropout': tune.choice([0.1, 0.2]),
    'decoder_layers': tune.choice([1, 2, 3, 4]),
    'scaler_type': tune.choice(['standard']),
    'learning_rate': tune.loguniform(1e-4, 1e-2),
    'batch_size': tune.choice([256, 512, 1024]),
    'encoder_hidden_size': tune.choice([128, 256, 512]),
    'decoder_hidden_size': tune.choice([128, 256, 512])
}

config_nhits = {
    "max_steps": 500,  # Number of SGD steps
    "input_size": tune.choice([5, 7, 14]),  # Size of input window
    "learning_rate": tune.loguniform(1e-5, 1e-1),  # Initial Learning rate
    # MaxPool's Kernelsize
    "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1], [64, 16, 1]]),
    # Interpolation expressivity ratios
    "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]),
    # Compute validation every 50 steps
    "val_check_steps": tune.choice([20, 30]),
    "dropout_prob_theta": tune.choice([0.2, 0.3, 0.4]),
    "batch_size": tune.choice([32, 64, 128, 256, 512]),
    "windows_batch_size": tune.choice([128, 256, 512])  # Random seed
}
