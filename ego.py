import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from collections import deque
import copy
from torch.optim.lr_scheduler import _LRScheduler

class AdaptiveLRScheduler(_LRScheduler):
    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, 
                 threshold_mode='rel', cooldown=0, min_lr=1e-8, max_lr=1.0, 
                 increase_factor=1.5, verbose=False):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.increase_factor = increase_factor
        self.verbose = verbose
        
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()
        super(AdaptiveLRScheduler, self).__init__(optimizer)

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics=None):
        # Raise warning if no metrics provided
        if metrics is None:
            if self.last_epoch != -1:
                warnings.warn("No metrics provided, AdaptiveLRScheduler cannot adjust learning rate.", UserWarning)
            return

        current = metrics
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.last_epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        elif self.num_bad_epochs == 0:
            self._increase_lr(self.last_epoch)

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.threshold:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}.')

    def _increase_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = min(old_lr * self.increase_factor, self.max_lr)
            if new_lr - old_lr > self.threshold:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Epoch {epoch}: increasing learning rate of group {i} to {new_lr:.4e}.')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -float('inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

class AdaptiveEarlyStopping:
    def __init__(self, patience=20, modification_delay=10):
        self.patience = patience
        self.modification_delay = modification_delay
        self.best_score = None
        self.counter = 0
        self.best_model = None
        self.last_modification_epoch = -float('inf')

    def __call__(self, model, epoch, val_loss, modifications):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
        elif score < self.best_score:
            if epoch - self.last_modification_epoch > self.modification_delay:
                self.counter += 1
                if self.counter >= self.patience:
                    return True, self.best_model
            else:
                self.counter = 0  # Reset counter if we're still in the modification delay period
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
            self.counter = 0

        if modifications:
            self.last_modification_epoch = epoch
            self.counter = 0  # Reset counter when modifications occur

        return False, self.best_model
    
class ArchitectureModificationPredictor(nn.Module):
    def __init__(self, input_features=5, hidden_size=64, num_modifications=5):
        super().__init__()
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_modifications)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

    def to(self, device):
        super().to(device)
        self.lstm.to(device)
        self.fc.to(device)
        return self
    
class AdaptiveModificationScheduler:
    def __init__(self, base_frequency=0.1, max_frequency=0.5, sensitivity=10):
        self.base_frequency = base_frequency
        self.max_frequency = max_frequency
        self.sensitivity = sensitivity
    
    def get_modification_probability(self, train_loss, eval_loss):
        loss_gap = max(0, eval_loss - train_loss)
        probability = self.base_frequency + (self.max_frequency - self.base_frequency) * (
            1 - math.exp(-self.sensitivity * loss_gap)
        )
        return min(probability, self.max_frequency)

class SelfAdaptingModificationScheduler:
    def __init__(self, model, initial_frequency=0.1, min_frequency=0.01, max_frequency=0.5, 
                 cooldown_period=5, performance_window=20, gap_threshold=0.5):
        self.model = model
        self.current_frequency = initial_frequency
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.cooldown_period = cooldown_period
        self.performance_window = performance_window
        self.gap_threshold = gap_threshold
        
        self.train_loss_history = []
        self.eval_loss_history = []
        self.modification_history = []
        self.last_modification_epoch = -1
        self.epoch = 0

    def should_modify(self):
        if self.epoch - self.last_modification_epoch < self.cooldown_period:
            return False
        return np.random.random() < self.current_frequency

    def update(self, train_loss, eval_loss):
        self.epoch += 1
        self.train_loss_history.append(train_loss)
        self.eval_loss_history.append(eval_loss)
        
        if len(self.train_loss_history) > self.performance_window:
            self.train_loss_history.pop(0)
            self.eval_loss_history.pop(0)
            self._adapt_frequency()

    def record_modification(self):
        self.last_modification_epoch = self.epoch
        self.modification_history.append(self.epoch)

    def _adapt_frequency(self):
        if len(self.train_loss_history) < 2:
            return

        recent_train_loss = np.mean(self.train_loss_history[-5:])
        recent_eval_loss = np.mean(self.eval_loss_history[-5:])
        recent_gap = recent_eval_loss - recent_train_loss

        previous_train_loss = np.mean(self.train_loss_history[:-5])
        previous_eval_loss = np.mean(self.eval_loss_history[:-5])
        previous_gap = previous_eval_loss - previous_train_loss

        gap_change = recent_gap - previous_gap

        if gap_change > self.gap_threshold:
            # Gap is growing, increase modification frequency
            self.current_frequency = min(self.current_frequency * 1.1, self.max_frequency)
        elif gap_change < -self.gap_threshold:
            # Gap is shrinking, decrease modification frequency
            self.current_frequency = max(self.current_frequency * 0.9, self.min_frequency)
        
        # Additional logic to prevent frequency from getting stuck at extremes
        if self.current_frequency == self.max_frequency and gap_change <= 0:
            self.current_frequency *= 0.95
        elif self.current_frequency == self.min_frequency and gap_change >= 0:
            self.current_frequency *= 1.05

    def get_stats(self):
        return {
            'frequency': self.current_frequency,
            'cooldown': self.cooldown_period,
            'modifications': len(self.modification_history),
            'parameters': self.model.count_parameters() if hasattr(self.model, 'count_parameters') else sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


class SelfModifyingNetwork(nn.Module):
    def __init__(self, input_channels=10, initial_hidden_channels=[64, 128], max_size_mb=100,
                 enable_layer_addition=True, enable_activation_modification=True, 
                 enable_forward_modification=True, device='cuda'):
        super(SelfModifyingNetwork, self).__init__()
        self._max_size_mb = max_size_mb
        self.device = device

        self.enable_layer_addition = enable_layer_addition
        self.enable_activation_modification = enable_activation_modification
        self.enable_forward_modification = enable_forward_modification

        self.onehot = nn.Embedding(input_channels, input_channels)
        self.onehot.weight.data = torch.eye(input_channels)
        self.onehot.weight.requires_grad = False

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        })
        self.current_activation = 'relu'

        self.encoder = nn.ModuleList([
            self.create_conv_block(input_channels, initial_hidden_channels[0]),
            self.create_conv_block(initial_hidden_channels[0], initial_hidden_channels[1]),
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.decoder = nn.ModuleList([
            self.create_conv_block(initial_hidden_channels[-1], 64),
            nn.Conv2d(64, input_channels, kernel_size=1)
        ])

        self.modification_history = []
        self.performance_history = []

        # New components for intelligent modification
        self.modification_predictor = ArchitectureModificationPredictor().to(self.device)
        self.modification_scheduler = AdaptiveModificationScheduler()
        self.to(self.device)

    def create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activations[self.current_activation]
        ).to(self.device) 

    @property
    def max_size_mb(self):
        return self._max_size_mb

    def forward(self, x):
        original_size = (x.shape[1], x.shape[2])
        
        x = x.to(self.device)

        x = self.onehot(x.long())
        x = x.permute(0, 3, 1, 2).float()
        
        for layer in self.encoder:
            x = layer(x)
        
        x = self.adaptive_pool(x)
        
        for layer in self.decoder[:-1]:
            x = layer(x)
        
        x = self.decoder[-1](x)
        
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x

    def should_modify(self, train_loss, eval_loss):
        probability = self.modification_scheduler.get_modification_probability(train_loss, eval_loss)
        return random.random() < probability

    def modify(self, train_loss, eval_loss):
        if not self.should_modify(train_loss, eval_loss):
            return []

        self.performance_history.append((train_loss, eval_loss))
        modifications = []

        # Use the modification predictor to decide which modification to make
        with torch.no_grad():
            performance_metrics = torch.tensor([train_loss, eval_loss, self.get_model_size(), len(self.encoder), len(self.decoder)]).unsqueeze(0).unsqueeze(0).to(self.device)
            modification_scores = self.modification_predictor(performance_metrics)
            modification_type = torch.argmax(modification_scores).item()

        if modification_type == 0 and self.enable_layer_addition and self.get_model_size() < self.max_size_mb * 0.9:
            new_channels = self.encoder[-1][0].out_channels
            new_layer = self.create_conv_block(new_channels, new_channels)
            self.encoder.append(new_layer)
            modifications.append(f"Added new encoder layer: Conv2d({new_channels}, {new_channels})")

        elif modification_type == 1 and self.enable_activation_modification:
            new_activation = random.choice(list(self.activations.keys()))
            if new_activation != self.current_activation:
                self.current_activation = new_activation
                for layer in self.encoder + self.decoder[:-1]:
                    if isinstance(layer, nn.Sequential):
                        layer[-1] = self.activations[self.current_activation]
                modifications.append(f"Changed activation to: {self.current_activation}")

        elif modification_type == 2 and self.enable_forward_modification:
            if len(self.encoder) > 2:
                skip_index = random.randint(0, len(self.encoder) - 2)
                self.encoder[skip_index].add_module('skip', nn.Identity())
                modifications.append(f"Added skip connection at layer {skip_index}")

        elif modification_type == 3:
            target_layer_index = random.randint(0, len(self.encoder) - 1)
            target_layer = self.encoder[target_layer_index]
            current_channels = target_layer[0].out_channels
            new_channels = max(32, current_channels + random.choice([-32, 32]))
            
            target_layer[0] = nn.Conv2d(target_layer[0].in_channels, new_channels, kernel_size=3, padding=1)
            target_layer[1] = nn.BatchNorm2d(new_channels)
            
            if target_layer_index < len(self.encoder) - 1:
                next_layer = self.encoder[target_layer_index + 1]
                next_layer[0] = nn.Conv2d(new_channels, next_layer[0].out_channels, kernel_size=3, padding=1)
            elif target_layer_index == len(self.encoder) - 1:
                self.decoder[0][0] = nn.Conv2d(new_channels, self.decoder[0][0].out_channels, kernel_size=3, padding=1)
            
            modifications.append(f"Adjusted channel count at layer {target_layer_index}: {current_channels} -> {new_channels}")

        if self.get_model_size() > self.max_size_mb:
            if modifications:
                modifications.pop()
                modifications.append("Reverted last modification due to size limit")

        self.modification_history.append((modification_type, modifications))
        
        if modifications:
            self.to(self.device)  # Move all parameters to the device after modifications
            
        return modifications
    
    def to(self, device):
        self.device = device
        return super().to(device)

    def get_model_size(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def train_modification_predictor(self):
        if len(self.modification_history) < 10:
            return  # Not enough data to train

        X = torch.tensor([[p[0], p[1], self.get_model_size(), len(self.encoder), len(self.decoder)] 
                          for p in self.performance_history[-len(self.modification_history):]])
        y = torch.tensor([h[0] for h in self.modification_history])
        
        optimizer = torch.optim.Adam(self.modification_predictor.parameters())
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(100):  # Adjust as needed
            optimizer.zero_grad()
            outputs = self.modification_predictor(X.unsqueeze(0))
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    def __setattr__(self, name, value):
        if name == 'max_size_mb':
            raise AttributeError("'max_size_mb' is read-only and cannot be modified.")
        super().__setattr__(name, value)