#!/usr/bin/env python3
"""
Neural Network Training Comparison Tool
Supports multiple gradient computation methods with time and memory tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.data import DataLoader
from typing import Union
from torch.nn.common_types import _size_2_t
import torch.optim as optim
import time
import json
import pickle
import os
import argparse
import psutil
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===== Model Implementations =====

class RGMLinearLayer(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, backward_dropout_rate, apply_reweighting=True):
        output = torch.matmul(input, weight.T) + bias
        ctx.save_for_backward(input, weight, bias)
        ctx.backward_dropout_rate = backward_dropout_rate
        ctx.apply_reweighting = apply_reweighting
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        backward_dropout_rate = ctx.backward_dropout_rate
        apply_reweighting = ctx.apply_reweighting
        dropout_rate = backward_dropout_rate

        # Create dropout mask, ensuring diagonal is always kept
        mask = torch.rand_like(weight) > dropout_rate
        mask.fill_diagonal_(True)

        if apply_reweighting:
            mask_weight = torch.ones_like(weight) / (1 - dropout_rate)
            mask_weight.fill_diagonal_(1.0)
            weighted_mask = mask_weight * mask
        else:
            mask_weight = torch.ones_like(weight)
            # mask_weight.fill_diagonal_(1.0)
            weighted_mask = mask_weight * mask

        grad_input = grad_output @ (weight * weighted_mask)
        grad_weight = grad_output.T @ input
        grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None


class RGMLinear(nn.Module):
    def __init__(self, in_features, out_features, backward_dropout_rate=0.0, apply_reweighting=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.backward_dropout_rate = backward_dropout_rate
        self.apply_reweighting = apply_reweighting
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return RGMLinearLayer.apply(x, self.weight, self.bias,
                                    self.backward_dropout_rate, self.apply_reweighting)


class SBPLinearFunction(Function):
    """
    Custom autograd function for Selective Backpropagation.
    Only computes gradients for selected samples, significantly reducing memory usage.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, keep_mask):
        """
        Forward pass: compute output for all samples but save context for selective backward.

        Args:
            input: Input tensor [batch_size, in_features]
            weight: Weight parameter [out_features, in_features]
            bias: Bias parameter [out_features] or None
            keep_mask: Boolean mask [batch_size] indicating which samples to keep
        """
        # Store tensors for backward pass
        ctx.save_for_backward(input, weight, bias, keep_mask)

        # Compute forward pass for all samples (memory efficient)
        output = F.linear(input, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients only for selected samples.
        This is where the major memory savings occur.
        """
        input, weight, bias, keep_mask = ctx.saved_tensors

        # Get indices of samples to keep
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]

        # Initialize gradient tensors
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        # Only compute gradients for kept samples
        if len(keep_indices) > 0:
            # Slice gradients and inputs for kept samples only
            grad_output_keep = grad_output[keep_indices]
            input_keep = input[keep_indices]

            # Compute gradients - memory usage scales with keep_ratio
            grad_input[keep_indices] = grad_output_keep @ weight
            grad_weight += grad_output_keep.T @ input_keep

            if bias is not None:
                grad_bias += grad_output_keep.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None


class SBPLinearBetter(nn.Module):
    """
    Memory-efficient Selective Backpropagation Linear Layer.

    Key optimizations:
    1. Custom autograd function for partial gradient computation
    2. Efficient mask generation and caching
    3. Minimal memory allocation overhead
    4. Proper gradient detachment for dropped samples
    """

    def __init__(self, in_features, out_features, bias=True, keep_ratio=0.5,
                 deterministic=False, device=None, dtype=None):
        """
        Initialize SBP Linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
            keep_ratio: Fraction of samples to keep for gradient computation
            deterministic: If True, use deterministic sample selection
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SBPLinearBetter, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.keep_ratio = keep_ratio
        self.deterministic = deterministic

        # Initialize parameters
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # Cache for deterministic mode
        self._cached_masks = {}

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _generate_keep_mask(self, batch_size, device):
        """
        Generate mask for sample selection with memory optimization.

        Args:
            batch_size: Size of current batch
            device: Device to place mask on

        Returns:
            Boolean mask indicating which samples to keep
        """
        if self.deterministic and batch_size in self._cached_masks:
            return self._cached_masks[batch_size].to(device)

        # Calculate number of samples to keep
        num_keep = max(1, int(batch_size * self.keep_ratio))

        # Create mask efficiently
        if self.deterministic:
            # Use deterministic selection for reproducibility
            torch.manual_seed(42)
            indices = torch.randperm(batch_size, device=device)[:num_keep]
            torch.manual_seed(torch.initial_seed())
        else:
            # Random selection for each forward pass
            indices = torch.randperm(batch_size, device=device)[:num_keep]

        # Create boolean mask
        keep_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        keep_mask[indices] = True

        # Cache for deterministic mode
        if self.deterministic:
            self._cached_masks[batch_size] = keep_mask.cpu()

        return keep_mask

    def forward(self, input):
        """
        Forward pass with selective backpropagation.

        Args:
            input: Input tensor [batch_size, in_features]

        Returns:
            Output tensor [batch_size, out_features]
        """
        batch_size = input.shape[0]

        # Generate keep mask
        keep_mask = self._generate_keep_mask(batch_size, input.device)

        # Use custom autograd function for memory-efficient computation
        return SBPLinearFunction.apply(input, self.weight, self.bias, keep_mask)

    def extra_repr(self):
        """String representation for debugging."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'keep_ratio={self.keep_ratio}, bias={self.bias is not None}'

    def get_memory_info(self, batch_size):
        """
        Get memory usage information for analysis.

        Args:
            batch_size: Batch size for calculation

        Returns:
            Dictionary with memory usage information
        """
        param_memory = (self.weight.numel() + (
            self.bias.numel() if self.bias is not None else 0)) * 4  # 4 bytes per float32

        # Memory usage scales with keep_ratio for gradients
        effective_batch_size = int(batch_size * self.keep_ratio)

        gradient_memory = param_memory  # Parameter gradients
        activation_memory = effective_batch_size * (self.in_features + self.out_features) * 4

        return {
            'parameter_memory_mb': param_memory / (1024 * 1024),
            'gradient_memory_mb': gradient_memory / (1024 * 1024),
            'activation_memory_mb': activation_memory / (1024 * 1024),
            'effective_batch_size': effective_batch_size,
            'memory_reduction_ratio': 1 - self.keep_ratio
        }


class SBPLinear2D(nn.Module):
    def __init__(self, in_features, out_features, bias=True, keep_ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.keep_ratio = keep_ratio

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _generate_keep_mask(self, batch_size):
        num_keep = int(batch_size * self.keep_ratio)
        keep_mask = torch.zeros(batch_size, dtype=torch.bool)
        indices = torch.randperm(batch_size)[:num_keep]
        keep_mask[indices] = True
        return keep_mask

    def forward(self, x):
        batch_size, in_features = x.shape
        keep_mask = self._generate_keep_mask(batch_size).to(x.device)

        # Memory optimization: use empty instead of zeros
        output = torch.empty(batch_size, self.out_features, device=x.device, dtype=x.dtype)

        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        drop_indices = (~keep_mask).nonzero(as_tuple=True)[0]

        # Compute forward pass for kept indices WITH gradient tracking
        if len(keep_indices) > 0:
            x_keep = x[keep_indices]
            out_keep = F.linear(x_keep, self.weight, self.bias)
            output[keep_indices] = out_keep

        # Compute forward pass for dropped indices WITHOUT gradient tracking
        if len(drop_indices) > 0:
            x_drop = x[drop_indices]
            with torch.no_grad():
                out_drop = F.linear(x_drop, self.weight, self.bias)
            output[drop_indices] = out_drop.detach()

        return output

############################################ Tiny Prop #################################################
class TinyPropParams:
    def __init__(self, S_min: float, S_max: float, zeta: float, number_of_layers: int):
        self.S_min = S_min
        self.S_max = S_max
        self.zeta = zeta
        self.number_of_layers = number_of_layers


class TinyPropLayer:
    def __init__(self, layerPosition: int):
        self.layerPosition = layerPosition
        self.Y_max = 0
        self.miniBatchBpr = 0
        self.miniBatchK = 0
        self.epochBpr = []
        self.epochK = []

    def BPR(self, params, Y):
        return (params.S_min + Y * (params.S_max - params.S_min) / (self.Y_max)) * (params.zeta ** self.layerPosition)

    def selectGradients(self, grad_output, params):
        # assumes grad_output.shape = [batchSize, entries]

        # calculate bpr (different across batches)
        Y = grad_output.abs().sum(1)  # Y [batchSize]
        if (torch.max(Y) > self.Y_max):  # Check if biggest Y of batch is bigger than recorded Y
            self.Y_max = torch.max(Y).item()
        bpr = self.BPR(params, Y)  # bpr [batchSize]

        # calculate K [batchSize]
        K = torch.round(grad_output.size(1) * bpr)  # K [batchSize]
        K.clamp(1, grad_output.size(1))
        self.miniBatchBpr += torch.mean(bpr).item()
        self.miniBatchK += torch.mean(K).item()
        K = K.to(torch.int16)

        # create a sparse grad_output tensor. Since k is different across batches, the topK indices
        # must be assembled for each batch separately.
        idx = []  # indices of sparse entries [batch, element]
        val = []  # corresponding values, of size element
        for batch, k in enumerate(K):
            _, indices = grad_output[batch].abs().topk(k)  # don't use return VALUES since they are abs!
            t = torch.vstack((torch.zeros_like(indices) + batch, indices))
            idx.append(t)
            val.append(torch.index_select(grad_output[batch], -1, indices))  # select values from grad_output instead

        idx = torch.hstack(idx)
        val = torch.cat(val)
        return idx, val


# ========== LINEAR ==========#

class SparseLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, tpParams: TinyPropParams, tpInfo: TinyPropLayer,
                bias=None):  # bias is an optional argument
        # Save inputs in context-object for later use in backwards. Alternatively, this part could be done in a setup_context() method
        ctx.save_for_backward(input, weight, bias)  # these are differentiable
        # non-differentiable arguments, directly stored on ctx
        ctx.tpParams = tpParams
        ctx.tpInfo = tpInfo

        # Do the mathematical operations associated with forwards
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Unpack saved tensors. NEVER modify these in the backwards function!
        input, weight, bias = ctx.saved_tensors
        # input   [batchSize, in]
        # output  [batchSize, out]
        # weights [out, in]
        # bias    [out]

        # Initialize all gradients w.r.t. inputs to None
        grad_input = grad_weight = grad_bias = None

        # This is the TinyProp part:
        indices, values = ctx.tpInfo.selectGradients(grad_output, ctx.tpParams)
        sparse = torch.sparse_coo_tensor(indices, values, grad_output.size())

        # Do the usual bp stuff but use sparse matmul on grad_input and grad_weight
        if ctx.needs_input_grad[0]:
            grad_input = torch.sparse.mm(sparse, weight)  # [batchSize, in]
        if ctx.needs_input_grad[1]:
            grad_weight = torch.sparse.mm(sparse.t(),
                                          input)  # Gradients are zeroed each batch, batch dimension is reduced in operation -> [out, in]
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, None, None, grad_bias


# Create TinyProp verion of Linear by extending it. This way it integrates seemlessly into existing code
class TinyPropLinear(TinyPropLayer, nn.Linear):
    def __init__(self, in_features: int, out_features: int, tinyPropParams: TinyPropParams, layer_number: int,
                 bias: bool = True, device=None, dtype=None):
        TinyPropLayer.__init__(self, tinyPropParams.number_of_layers - layer_number)
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)

        # Saving variables like this will pass it by REFERENCE, so changes
        # made in backwards are reflected in layer
        self.tpParams = tinyPropParams

    def forward(self, input):
        # Here the custom linear function is applied
        return SparseLinear.apply(input, self.weight, self.tpParams, self, self.bias)



################################################ Tiny Prop ################################################


class SimpleNet(nn.Module):
    def __init__(self,
                 input_dim=196,
                 output_dim=10,
                 hidden_dims=[128, 64, 32],
                 layer_type='standard',
                 keep_ratio=0.5,
                 activation='relu'):
        super(SimpleNet, self).__init__()
        self.layer_type = layer_type
        self.input_dim = input_dim

        self.layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(self.layer_dims) - 1):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]

            if layer_type == 'standard':
                layer = nn.Linear(in_dim, out_dim)
            elif layer_type == 'rgm':
                layer = RGMLinear(in_dim, out_dim, backward_dropout_rate=1 - keep_ratio)
            elif layer_type == 'rgm-no-reweighting':
                layer = RGMLinear(in_dim, out_dim, backward_dropout_rate=1 - keep_ratio, apply_reweighting=False)
            elif layer_type == 'sbp':
                # layer = SBPLinear2D(in_dim, out_dim, keep_ratio=keep_ratio)
                layer = SBPLinearBetter(in_dim, out_dim, keep_ratio=keep_ratio)
            # elif layer_type == 'tinyprop':
            #     layer = TinyPropLinear(in_dim, out_dim, keep_ratio=keep_ratio)
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")

            self.layers.append(layer)

        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def get_architecture_info(self):
        return {
            'layer_dims': self.layer_dims,
            'layer_type': self.layer_type,
            'num_layers': len(self.layers),
            'total_params': sum(p.numel() for p in self.parameters())
        }


# ===== Training and Evaluation Functions =====

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return 100. * correct / total


def train_model(model, optimizer, train_loader, test_loader, num_epochs, device, model_name):
    """Train a single model with comprehensive time and memory tracking."""

    # Initialize tracking
    start_time = time.time()
    start_memory = get_memory_usage()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        start_gpu_memory = torch.cuda.memory_allocated(device) / 1024 / 1024
    else:
        start_gpu_memory = 0

    model.train()
    criterion = nn.CrossEntropyLoss()

    # Storage for metrics
    epoch_losses = []
    epoch_test_accuracies = []
    epoch_times = []
    memory_usage = []
    gpu_memory_usage = []

    logger.info(f"Starting training for {model_name}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        # Record epoch metrics
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # Memory tracking
        current_memory = get_memory_usage()
        memory_usage.append(current_memory)

        if device.type == 'cuda':
            current_gpu_memory = torch.cuda.memory_allocated(device) / 1024 / 1024
            gpu_memory_usage.append(current_gpu_memory)

        # Test accuracy
        test_acc = evaluate_model(model, test_loader, device)
        epoch_test_accuracies.append(test_acc)

        if (epoch + 1) % 5 == 0:
            train_acc = 100. * correct / total
            logger.info(f"Epoch {epoch + 1:2d}/{num_epochs}: Loss={avg_loss:.4f}, "
                        f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    # Final calculations
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    memory_increase = final_memory - start_memory

    if device.type == 'cuda':
        peak_gpu_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        gpu_memory_increase = peak_gpu_memory - start_gpu_memory
    else:
        peak_gpu_memory = 0
        gpu_memory_increase = 0

    final_test_acc = epoch_test_accuracies[-1]

    logger.info(f"✅ {model_name} completed - Test Acc: {final_test_acc:.2f}%, "
                f"Time: {total_time:.2f}s, Memory: {memory_increase:.2f}MB increase")

    return {
        'model_name': model_name,
        'test_accuracy': final_test_acc,
        'epoch_losses': epoch_losses,
        'epoch_test_accuracies': epoch_test_accuracies,
        'training_time_seconds': total_time,
        'training_time_minutes': total_time / 60,
        'epoch_times': epoch_times,
        'memory_usage': {
            'start_memory_mb': start_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'peak_memory_mb': max(memory_usage),
            'epoch_memory_usage': memory_usage
        },
        'gpu_memory_usage': {
            'start_gpu_memory_mb': start_gpu_memory,
            'peak_gpu_memory_mb': peak_gpu_memory,
            'gpu_memory_increase_mb': gpu_memory_increase,
            'epoch_gpu_memory_usage': gpu_memory_usage
        } if device.type == 'cuda' else None
    }


def create_optimizer(model_parameters, config):
    """Create optimizer based on configuration."""
    optimizer_name = config['optimizer'].lower()
    base_params = {'lr': config['learning_rate']}
    base_params.update(config.get('optimizer_params', {}))

    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adamw': optim.AdamW
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported. Use: {list(optimizers.keys())}")

    return optimizers[optimizer_name](model_parameters, **base_params)


def train_selected_models(config):
    """Train only selected models based on configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"🚀 Starting Training on {device}")
    logger.info("=" * 60)

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(config['normalize_mean'], config['normalize_std'])
    ])

    if config['dataset'] == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset {config['dataset']} not supported")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], shuffle=False)

    # Results storage
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'models': {}
    }

    def clear_memory():
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    # Train selected models
    for model_type in config['models_to_train']:
        if model_type == 'baseline':
            logger.info("🔧 Training Baseline Model...")
            clear_memory()
            model = SimpleNet(layer_type='standard').to(device)
            optimizer = create_optimizer(model.parameters(), config)

            result = train_model(model, optimizer, train_loader, test_loader,
                                 config['num_epochs'], device, "Baseline")
            results['models']['baseline'] = result

        elif model_type == 'rgm':
            for keep_ratio in config['keep_ratios']:
                logger.info(f"🔧 Training RGM Model (keep_ratio={keep_ratio})...")
                clear_memory()
                model = SimpleNet(layer_type='rgm', keep_ratio=keep_ratio).to(device)
                optimizer = create_optimizer(model.parameters(), config)

                result = train_model(model, optimizer, train_loader, test_loader,
                                     config['num_epochs'], device, f"RGM-{keep_ratio}")

                if 'rgm' not in results['models']:
                    results['models']['rgm'] = {}
                results['models']['rgm'][keep_ratio] = result

        elif model_type == 'sbp':
            for keep_ratio in config['keep_ratios']:
                logger.info(f"🔧 Training SBP Model (keep_ratio={keep_ratio})...")
                clear_memory()
                model = SimpleNet(layer_type='sbp', keep_ratio=keep_ratio).to(device)
                optimizer = create_optimizer(model.parameters(), config)

                result = train_model(model, optimizer, train_loader, test_loader,
                                     config['num_epochs'], device, f"SBP-{keep_ratio}")

                if 'sbp' not in results['models']:
                    results['models']['sbp'] = {}
                results['models']['sbp'][keep_ratio] = result

    return results


def save_results(results, output_dir='./results'):
    """Save results to JSON and pickle files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON (human readable)
    json_path = os.path.join(output_dir, f'training_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save as pickle (preserves all data types)
    pickle_path = os.path.join(output_dir, f'training_results_{timestamp}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"Results saved to:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  Pickle: {pickle_path}")

    return json_path, pickle_path


def create_comparison_plots(results, output_dir='./results'):
    """Create and save comparison plots."""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    # Extract data for plotting
    models = results['models']
    keep_ratios = results['config']['keep_ratios']

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Test Accuracy Comparison
    ax1 = axes[0, 0]
    if 'baseline' in models:
        baseline_acc = [models['baseline']['test_accuracy']] * len(keep_ratios)
        ax1.plot(keep_ratios, baseline_acc, 'o-', label='Baseline', linewidth=2)

    if 'rgm' in models:
        rgm_acc = [models['rgm'][kr]['test_accuracy'] for kr in keep_ratios]
        ax1.plot(keep_ratios, rgm_acc, 's-', label='RGM', linewidth=2)

    if 'sbp' in models:
        sbp_acc = [models['sbp'][kr]['test_accuracy'] for kr in keep_ratios]
        ax1.plot(keep_ratios, sbp_acc, '^-', label='SBP', linewidth=2)

    ax1.set_xlabel('Keep Ratio')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Time Comparison
    ax2 = axes[0, 1]
    if 'baseline' in models:
        baseline_time = [models['baseline']['training_time_seconds']] * len(keep_ratios)
        ax2.plot(keep_ratios, baseline_time, 'o-', label='Baseline', linewidth=2)

    if 'rgm' in models:
        rgm_time = [models['rgm'][kr]['training_time_seconds'] for kr in keep_ratios]
        ax2.plot(keep_ratios, rgm_time, 's-', label='RGM', linewidth=2)

    if 'sbp' in models:
        sbp_time = [models['sbp'][kr]['training_time_seconds'] for kr in keep_ratios]
        ax2.plot(keep_ratios, sbp_time, '^-', label='SBP', linewidth=2)

    ax2.set_xlabel('Keep Ratio')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Memory Usage Comparison
    ax3 = axes[1, 0]
    if 'baseline' in models:
        baseline_mem = [models['baseline']['memory_usage']['memory_increase_mb']] * len(keep_ratios)
        ax3.plot(keep_ratios, baseline_mem, 'o-', label='Baseline', linewidth=2)

    if 'rgm' in models:
        rgm_mem = [models['rgm'][kr]['memory_usage']['memory_increase_mb'] for kr in keep_ratios]
        ax3.plot(keep_ratios, rgm_mem, 's-', label='RGM', linewidth=2)

    if 'sbp' in models:
        sbp_mem = [models['sbp'][kr]['memory_usage']['memory_increase_mb'] for kr in keep_ratios]
        ax3.plot(keep_ratios, sbp_mem, '^-', label='SBP', linewidth=2)

    ax3.set_xlabel('Keep Ratio')
    ax3.set_ylabel('Memory Increase (MB)')
    ax3.set_title('Memory Usage Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: GPU Memory Usage (if available)
    ax4 = axes[1, 1]
    if results['device'].startswith('cuda'):
        if 'baseline' in models and models['baseline']['gpu_memory_usage']:
            baseline_gpu = [models['baseline']['gpu_memory_usage']['gpu_memory_increase_mb']] * len(keep_ratios)
            ax4.plot(keep_ratios, baseline_gpu, 'o-', label='Baseline', linewidth=2)

        if 'rgm' in models:
            rgm_gpu = [models['rgm'][kr]['gpu_memory_usage']['gpu_memory_increase_mb'] for kr in keep_ratios]
            ax4.plot(keep_ratios, rgm_gpu, 's-', label='RGM', linewidth=2)

        if 'sbp' in models:
            sbp_gpu = [models['sbp'][kr]['gpu_memory_usage']['gpu_memory_increase_mb'] for kr in keep_ratios]
            ax4.plot(keep_ratios, sbp_gpu, '^-', label='SBP', linewidth=2)

        ax4.set_xlabel('Keep Ratio')
        ax4.set_ylabel('GPU Memory Increase (MB)')
        ax4.set_title('GPU Memory Usage Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'GPU not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('GPU Memory Usage (N/A)')

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'comparison_plots_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plots saved to: {plot_path}")

    plt.show()
    return plot_path


def main():
    parser = argparse.ArgumentParser(description='Neural Network Training Comparison Tool')
    parser.add_argument('--models', nargs='+', default=['baseline'],
                        choices=['baseline', 'rgm', 'sbp'],
                        help='Models to train (default: all)')
    parser.add_argument('--keep-ratios', nargs='+', type=float, default=[0.5],
                        help='Keep ratios to test (default: 0.3 0.5 0.7)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size (default: 1024)')
    parser.add_argument('--optimizer', default='sgd',
                        choices=['adam', 'sgd', 'rmsprop', 'adamw'],
                        help='Optimizer (default: adam)')
    parser.add_argument('--output-dir', default='./results',
                        help='Output directory for results (default: ./results)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip creating comparison plots')

    args = parser.parse_args()

    # Configuration
    config = {
        'models_to_train': args.models,
        'keep_ratios': args.keep_ratios,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size_train': args.batch_size,
        'batch_size_test': args.batch_size,
        'optimizer': args.optimizer,
        'optimizer_params': {},
        'dataset': 'MNIST',
        'image_size': (14, 14),
        'normalize_mean': (0.1307,),
        'normalize_std': (0.3081,)
    }

    logger.info(f"Configuration: {config}")

    # Train models
    results = train_selected_models(config)

    # Save results
    json_path, pickle_path = save_results(results, args.output_dir)

    # Create plots
    if not args.no_plots:
        plot_path = create_comparison_plots(results, args.output_dir)

    logger.info("Training completed successfully!")
    logger.info(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
