"""
GPU CNN Trainer - Full Backpropagation Training Loop

Implements end-to-end GPU training for DeepSeek CNN:
- Forward pass (GPU)
- Loss computation (cross-entropy)
- Backward pass (GPU gradients)
- Weight updates (SGD with momentum)
"""

from __future__ import annotations

import numpy as np
import ctypes
from pathlib import Path
from typing import Dict, List, Tuple

from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
from knowledge3d.cranium.ocr.gpu_backward import GPUBackward
from knowledge3d.cranium.sovereign import loader


class GPUCNNTrainer:
    """Full GPU training pipeline for CNN."""

    def __init__(
        self,
        model: DeepSeekOCRModel,
        num_classes: int = 62,  # A-Z, a-z, 0-9
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        normalize_gradients: bool = True,
        fc_only: bool = False,
    ):
        """
        Initialize GPU trainer.

        Args:
            model: DeepSeek OCR model (forward pass)
            num_classes: Number of character classes
            learning_rate: SGD learning rate
            momentum: SGD momentum coefficient
        """
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.normalize_gradients = normalize_gradients
        self.fc_only = fc_only

        # Initialize backward pass
        self.gpu_backward = GPUBackward()

        # Allocate gradient buffers
        self._allocate_gradients()

        # Allocate velocity buffers for momentum
        self._allocate_velocities()

        # FC layer for classification
        self.fc_weight = np.random.randn(num_classes, 128).astype(np.float32) * 0.01
        self.fc_bias = np.zeros(num_classes, dtype=np.float32)

        # Upload FC to GPU
        self.d_fc_weight = loader.gpu_malloc(self.fc_weight.nbytes)
        self.d_fc_bias = loader.gpu_malloc(self.fc_bias.nbytes)
        loader.memcpy_htod(self.d_fc_weight, self.fc_weight.ctypes.data_as(ctypes.c_void_p), self.fc_weight.nbytes)
        loader.memcpy_htod(self.d_fc_bias, self.fc_bias.ctypes.data_as(ctypes.c_void_p), self.fc_bias.nbytes)

        # Gradient monitoring configuration
        self._batch_counter = 0
        self.gradient_log_interval = 10

    def _allocate_gradients(self):
        """Allocate GPU memory for all gradients."""
        # Conv1 gradients
        self.d_grad_conv1_w = self.gpu_backward.allocate_grad_buffer(
            "conv1_weight", self.model.conv1_weight.shape
        )
        self.d_grad_conv1_b = self.gpu_backward.allocate_grad_buffer(
            "conv1_bias", self.model.conv1_bias.shape
        )

        # BN1 gradients
        self.d_grad_bn1_gamma = self.gpu_backward.allocate_grad_buffer(
            "bn1_gamma", self.model.bn1_gamma.shape
        )
        self.d_grad_bn1_beta = self.gpu_backward.allocate_grad_buffer(
            "bn1_beta", self.model.bn1_beta.shape
        )

        # Conv2 gradients
        self.d_grad_conv2_w = self.gpu_backward.allocate_grad_buffer(
            "conv2_weight", self.model.conv2_weight.shape
        )
        self.d_grad_conv2_b = self.gpu_backward.allocate_grad_buffer(
            "conv2_bias", self.model.conv2_bias.shape
        )

        # BN2 gradients
        self.d_grad_bn2_gamma = self.gpu_backward.allocate_grad_buffer(
            "bn2_gamma", self.model.bn2_gamma.shape
        )
        self.d_grad_bn2_beta = self.gpu_backward.allocate_grad_buffer(
            "bn2_beta", self.model.bn2_beta.shape
        )

        # Conv3 gradients
        self.d_grad_conv3_w = self.gpu_backward.allocate_grad_buffer(
            "conv3_weight", self.model.conv3_weight.shape
        )
        self.d_grad_conv3_b = self.gpu_backward.allocate_grad_buffer(
            "conv3_bias", self.model.conv3_bias.shape
        )

        # BN3 gradients
        self.d_grad_bn3_gamma = self.gpu_backward.allocate_grad_buffer(
            "bn3_gamma", self.model.bn3_gamma.shape
        )
        self.d_grad_bn3_beta = self.gpu_backward.allocate_grad_buffer(
            "bn3_beta", self.model.bn3_beta.shape
        )

        # FC gradients
        self.d_grad_fc_weight = self.gpu_backward.allocate_grad_buffer(
            "fc_weight", (self.num_classes, 128)
        )
        self.d_grad_fc_bias = self.gpu_backward.allocate_grad_buffer(
            "fc_bias", (self.num_classes,)
        )

    def _allocate_velocities(self):
        """Allocate GPU memory for momentum velocities."""
        # Conv1 velocity
        self.d_vel_conv1_w = self.gpu_backward.allocate_grad_buffer(
            "vel_conv1_weight", self.model.conv1_weight.shape
        )
        self.d_vel_conv1_b = self.gpu_backward.allocate_grad_buffer(
            "vel_conv1_bias", self.model.conv1_bias.shape
        )

        # BN1 velocity
        self.d_vel_bn1_gamma = self.gpu_backward.allocate_grad_buffer(
            "vel_bn1_gamma", self.model.bn1_gamma.shape
        )
        self.d_vel_bn1_beta = self.gpu_backward.allocate_grad_buffer(
            "vel_bn1_beta", self.model.bn1_beta.shape
        )

        # Conv2 velocity
        self.d_vel_conv2_w = self.gpu_backward.allocate_grad_buffer(
            "vel_conv2_weight", self.model.conv2_weight.shape
        )
        self.d_vel_conv2_b = self.gpu_backward.allocate_grad_buffer(
            "vel_conv2_bias", self.model.conv2_bias.shape
        )

        # BN2 velocity
        self.d_vel_bn2_gamma = self.gpu_backward.allocate_grad_buffer(
            "vel_bn2_gamma", self.model.bn2_gamma.shape
        )
        self.d_vel_bn2_beta = self.gpu_backward.allocate_grad_buffer(
            "vel_bn2_beta", self.model.bn2_beta.shape
        )

        # Conv3 velocity
        self.d_vel_conv3_w = self.gpu_backward.allocate_grad_buffer(
            "vel_conv3_weight", self.model.conv3_weight.shape
        )
        self.d_vel_conv3_b = self.gpu_backward.allocate_grad_buffer(
            "vel_conv3_bias", self.model.conv3_bias.shape
        )

        # BN3 velocity
        self.d_vel_bn3_gamma = self.gpu_backward.allocate_grad_buffer(
            "vel_bn3_gamma", self.model.bn3_gamma.shape
        )
        self.d_vel_bn3_beta = self.gpu_backward.allocate_grad_buffer(
            "vel_bn3_beta", self.model.bn3_beta.shape
        )

        # FC velocity
        self.d_vel_fc_weight = self.gpu_backward.allocate_grad_buffer(
            "vel_fc_weight", (self.num_classes, 128)
        )
        self.d_vel_fc_bias = self.gpu_backward.allocate_grad_buffer(
            "vel_fc_bias", (self.num_classes,)
        )

    def forward(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass through CNN + FC layer.

        Args:
            image: Input image [H, W, 3], float32 in [0, 1]

        Returns:
            logits: Class logits [num_classes]
            probs: Softmax probabilities [num_classes]
            cache: Intermediate activations for backward pass
        """
        # CNN forward pass with activation caching
        result = self.model.forward(image, cache_for_backward=True)
        cache = result.get('cache', {})
        feature_map = cache.get('conv3_out', result['feature_map'])  # Use pre-BN activations when available

        # Global average pooling
        features = feature_map.mean(axis=(0, 1))  # [128]

        # FC layer
        logits = self.fc_weight @ features + self.fc_bias  # [num_classes]

        # Softmax
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()

        # Cache for backward
        cache = {
            'feature_map': feature_map,
            'features': features,
            'logits': logits,
            'probs': probs,
            # Also cache intermediate CNN activations from model
            **cache,
        }

        return logits, probs, cache

    def accumulate_gradients(self, image: np.ndarray, target: int, cache: Dict) -> float:
        """
        Backward pass: compute and accumulate gradients (NO weight updates).

        CRITICAL FIX: This method only accumulates gradients across samples.
        Weight updates happen in _update_all_weights() after full batch processing.

        Args:
            image: Input image [H, W, 3]
            target: True class index
            cache: Activations from forward pass

        Returns:
            loss: Cross-entropy loss value
        """
        probs = cache['probs']
        features = cache['features']  # [128]
        feature_map = cache['feature_map']  # [H, W, 128]

        # Compute loss
        prob_target = np.clip(probs[target], 1e-7, 1.0)
        loss = -np.log(prob_target)

        # Gradient of loss w.r.t. logits (softmax + cross-entropy)
        d_logits = probs.copy()  # [num_classes]
        d_logits[target] -= 1.0

        # Gradient w.r.t. FC weights and bias
        d_fc_weight = np.outer(d_logits, features)  # [num_classes, 128]
        d_fc_bias = d_logits  # [num_classes]

        # Gradient w.r.t. features
        d_features = self.fc_weight.T @ d_logits  # [128]

        # Gradient w.r.t. feature map (reverse global avg pool)
        H, W, C = feature_map.shape
        d_feature_map = np.tile(d_features / (H * W), (H, W, 1))  # [H, W, 128]

        # Accumulate FC gradients on GPU immediately so we can short-circuit for fc-only runs
        d_fc_weight_gpu = loader.gpu_malloc(d_fc_weight.nbytes)
        d_fc_bias_gpu = loader.gpu_malloc(d_fc_bias.nbytes)
        loader.memcpy_htod(d_fc_weight_gpu, d_fc_weight.ctypes.data_as(ctypes.c_void_p), d_fc_weight.nbytes)
        loader.memcpy_htod(d_fc_bias_gpu, d_fc_bias.ctypes.data_as(ctypes.c_void_p), d_fc_bias.nbytes)

        fc_weight_grad_acc = np.empty_like(d_fc_weight)
        fc_bias_grad_acc = np.empty_like(d_fc_bias)
        loader.memcpy_dtoh(fc_weight_grad_acc.ctypes.data_as(ctypes.c_void_p), self.d_grad_fc_weight, d_fc_weight.nbytes)
        loader.memcpy_dtoh(fc_bias_grad_acc.ctypes.data_as(ctypes.c_void_p), self.d_grad_fc_bias, d_fc_bias.nbytes)
        fc_weight_grad_acc += d_fc_weight
        fc_bias_grad_acc += d_fc_bias
        loader.memcpy_htod(self.d_grad_fc_weight, fc_weight_grad_acc.ctypes.data_as(ctypes.c_void_p), d_fc_weight.nbytes)
        loader.memcpy_htod(self.d_grad_fc_bias, fc_bias_grad_acc.ctypes.data_as(ctypes.c_void_p), d_fc_bias.nbytes)

        loader.gpu_free(d_fc_weight_gpu)
        loader.gpu_free(d_fc_bias_gpu)

        if self.fc_only:
            return loss

        # ================================================================
        # FULL CNN BACKWARD PASS - GPU ONLY
        # ================================================================
        # Gradient flow: FC → AvgPool → BN3 → Conv3 → (continuing...)

        # Cached activations are directly in cache (unpacked in forward method)

        # ----------------------------------------------------------------
        # BN3 backward: Proper BatchNorm backward pass
        # ----------------------------------------------------------------
        # d_feature_map is gradient w.r.t. BN3 output [16, 16, 128]
        conv3_out = cache['conv3_out']  # Input to BN3 [16, 16, 128]
        bn3_mean = cache['bn3_mean']    # [128]
        bn3_var = cache['bn3_var']      # [128]

        # Allocate gradient output buffer for BN3 input
        d_grad_conv3_out_gpu = loader.gpu_malloc(conv3_out.nbytes)

        # Call BatchNorm backward kernel
        self.gpu_backward.batchnorm_backward_training(
            d_feature_map,
            cache['bn3_x_hat'],
            self.model.bn3_gamma,
            bn3_mean,
            bn3_var,
            d_grad_conv3_out_gpu,
            self.d_grad_bn3_gamma,
            self.d_grad_bn3_beta,
            eps=1e-5,
        )

        # Download gradient w.r.t. Conv3 output
        d_grad_conv3_out = np.empty_like(conv3_out)
        loader.memcpy_dtoh(d_grad_conv3_out.ctypes.data_as(ctypes.c_void_p), d_grad_conv3_out_gpu, conv3_out.nbytes)
        loader.gpu_free(d_grad_conv3_out_gpu)

        # ----------------------------------------------------------------
        # Conv3 backward: [16x16x64] → [16x16x128]
        # ----------------------------------------------------------------
        bn2_out = cache['bn2_out']  # [16, 16, 64]

        # Apply ReLU backward (zero out gradients where input was < 0)
        # RPN-style NaN guard: sanitize gradients before ReLU backward
        d_grad_conv3_out = np.nan_to_num(d_grad_conv3_out, nan=0.0, posinf=0.0, neginf=0.0)
        d_grad_conv3_pre_relu = d_grad_conv3_out * (conv3_out > 0)

        # Pad bn2_out for conv backward
        bn2_out_padded = np.pad(bn2_out, ((1, 1), (1, 1), (0, 0)), mode='constant')

        # Compute Conv3 weight/bias gradients
        self.gpu_backward.conv_backward_weight(
            d_grad_conv3_pre_relu, bn2_out_padded,
            self.d_grad_conv3_w, self.d_grad_conv3_b,
            Cin=64, Cout=128
        )

        # Conv3 input gradient
        H_bn2, W_bn2 = bn2_out.shape[:2]
        d_grad_bn2_padded = loader.gpu_malloc(bn2_out_padded.nbytes)
        self.gpu_backward.conv_backward_input(
            d_grad_conv3_pre_relu, self.model.conv3_weight,
            d_grad_bn2_padded, H_bn2, W_bn2, Cin=64, Cout=128
        )

        # Extract inner region (remove padding)
        grad_bn2_padded = np.empty_like(bn2_out_padded)
        loader.memcpy_dtoh(grad_bn2_padded.ctypes.data_as(ctypes.c_void_p), d_grad_bn2_padded, bn2_out_padded.nbytes)
        d_grad_bn2_out = grad_bn2_padded[1:-1, 1:-1, :]
        loader.gpu_free(d_grad_bn2_padded)

        # ----------------------------------------------------------------
        # BN2 backward
        # ----------------------------------------------------------------
        pool2_out = cache['pool2_out']
        bn2_mean = cache['bn2_mean']
        bn2_var = cache['bn2_var']

        d_grad_pool2_out_gpu = loader.gpu_malloc(pool2_out.nbytes)
        self.gpu_backward.batchnorm_backward_training(
            d_grad_bn2_out,
            cache['bn2_x_hat'],
            self.model.bn2_gamma,
            bn2_mean,
            bn2_var,
            d_grad_pool2_out_gpu,
            self.d_grad_bn2_gamma,
            self.d_grad_bn2_beta,
        )

        d_grad_pool2_out = np.empty_like(pool2_out)
        loader.memcpy_dtoh(d_grad_pool2_out.ctypes.data_as(ctypes.c_void_p), d_grad_pool2_out_gpu, pool2_out.nbytes)
        loader.gpu_free(d_grad_pool2_out_gpu)

        # ----------------------------------------------------------------
        # Pool2 backward
        # ----------------------------------------------------------------
        conv2_out = cache['conv2_out']
        d_grad_conv2_out_gpu = loader.gpu_malloc(conv2_out.nbytes)
        self.gpu_backward.maxpool_backward(
            d_grad_pool2_out, conv2_out, d_grad_conv2_out_gpu
        )

        d_grad_conv2_out = np.empty_like(conv2_out)
        loader.memcpy_dtoh(d_grad_conv2_out.ctypes.data_as(ctypes.c_void_p), d_grad_conv2_out_gpu, conv2_out.nbytes)
        loader.gpu_free(d_grad_conv2_out_gpu)

        # ----------------------------------------------------------------
        # Conv2 backward
        # ----------------------------------------------------------------
        # RPN-style NaN guard: sanitize gradients before ReLU backward
        d_grad_conv2_out = np.nan_to_num(d_grad_conv2_out, nan=0.0, posinf=0.0, neginf=0.0)
        d_grad_conv2_pre_relu = d_grad_conv2_out * (conv2_out > 0)
        bn1_out = cache['bn1_out']
        bn1_out_padded = np.pad(bn1_out, ((1, 1), (1, 1), (0, 0)), mode='constant')

        self.gpu_backward.conv_backward_weight(
            d_grad_conv2_pre_relu, bn1_out_padded,
            self.d_grad_conv2_w, self.d_grad_conv2_b,
            Cin=32, Cout=64
        )

        H_bn1, W_bn1 = bn1_out.shape[:2]
        d_grad_bn1_padded = loader.gpu_malloc(bn1_out_padded.nbytes)
        self.gpu_backward.conv_backward_input(
            d_grad_conv2_pre_relu, self.model.conv2_weight,
            d_grad_bn1_padded, H_bn1, W_bn1, Cin=32, Cout=64
        )

        grad_bn1_padded = np.empty_like(bn1_out_padded)
        loader.memcpy_dtoh(grad_bn1_padded.ctypes.data_as(ctypes.c_void_p), d_grad_bn1_padded, bn1_out_padded.nbytes)
        d_grad_bn1_out = grad_bn1_padded[1:-1, 1:-1, :]
        loader.gpu_free(d_grad_bn1_padded)

        # ----------------------------------------------------------------
        # BN1 backward
        # ----------------------------------------------------------------
        pool1_out = cache['pool1_out']
        bn1_mean = cache['bn1_mean']
        bn1_var = cache['bn1_var']

        d_grad_pool1_out_gpu = loader.gpu_malloc(pool1_out.nbytes)
        self.gpu_backward.batchnorm_backward_training(
            d_grad_bn1_out,
            cache['bn1_x_hat'],
            self.model.bn1_gamma,
            bn1_mean,
            bn1_var,
            d_grad_pool1_out_gpu,
            self.d_grad_bn1_gamma,
            self.d_grad_bn1_beta,
        )

        d_grad_pool1_out = np.empty_like(pool1_out)
        loader.memcpy_dtoh(d_grad_pool1_out.ctypes.data_as(ctypes.c_void_p), d_grad_pool1_out_gpu, pool1_out.nbytes)
        loader.gpu_free(d_grad_pool1_out_gpu)

        # ----------------------------------------------------------------
        # Pool1 backward
        # ----------------------------------------------------------------
        conv1_out = cache['conv1_out']
        d_grad_conv1_out_gpu = loader.gpu_malloc(conv1_out.nbytes)
        self.gpu_backward.maxpool_backward(
            d_grad_pool1_out, conv1_out, d_grad_conv1_out_gpu
        )

        d_grad_conv1_out = np.empty_like(conv1_out)
        loader.memcpy_dtoh(d_grad_conv1_out.ctypes.data_as(ctypes.c_void_p), d_grad_conv1_out_gpu, conv1_out.nbytes)
        loader.gpu_free(d_grad_conv1_out_gpu)

        # ----------------------------------------------------------------
        # Conv1 backward
        # ----------------------------------------------------------------
        # RPN-style NaN guard: sanitize gradients before ReLU backward
        d_grad_conv1_out = np.nan_to_num(d_grad_conv1_out, nan=0.0, posinf=0.0, neginf=0.0)
        d_grad_conv1_pre_relu = d_grad_conv1_out * (conv1_out > 0)
        input_img = cache['input']
        input_padded = np.pad(input_img, ((1, 1), (1, 1), (0, 0)), mode='constant')

        self.gpu_backward.conv_backward_weight(
            d_grad_conv1_pre_relu, input_padded,
            self.d_grad_conv1_w, self.d_grad_conv1_b,
            Cin=3, Cout=32
        )

        # Full backward chain complete! Gradients accumulated on GPU.
        # NO weight updates here - they happen in _update_all_weights() after batch processing.

        return loss

    def _zero_gradients(self):
        """Zero all gradient buffers before batch processing."""
        # Zero CNN gradients
        self.gpu_backward.zero_gradients(self.d_grad_conv1_w, self.model.conv1_weight.size)
        self.gpu_backward.zero_gradients(self.d_grad_conv1_b, self.model.conv1_bias.size)
        self.gpu_backward.zero_gradients(self.d_grad_bn1_gamma, self.model.bn1_gamma.size)
        self.gpu_backward.zero_gradients(self.d_grad_bn1_beta, self.model.bn1_beta.size)

        self.gpu_backward.zero_gradients(self.d_grad_conv2_w, self.model.conv2_weight.size)
        self.gpu_backward.zero_gradients(self.d_grad_conv2_b, self.model.conv2_bias.size)
        self.gpu_backward.zero_gradients(self.d_grad_bn2_gamma, self.model.bn2_gamma.size)
        self.gpu_backward.zero_gradients(self.d_grad_bn2_beta, self.model.bn2_beta.size)

        self.gpu_backward.zero_gradients(self.d_grad_conv3_w, self.model.conv3_weight.size)
        self.gpu_backward.zero_gradients(self.d_grad_conv3_b, self.model.conv3_bias.size)
        self.gpu_backward.zero_gradients(self.d_grad_bn3_gamma, self.model.bn3_gamma.size)
        self.gpu_backward.zero_gradients(self.d_grad_bn3_beta, self.model.bn3_beta.size)

        # Zero FC gradients
        self.gpu_backward.zero_gradients(self.d_grad_fc_weight, self.fc_weight.size)
        self.gpu_backward.zero_gradients(self.d_grad_fc_bias, self.fc_bias.size)

    def _scale_gradients(self, batch_size: int):
        """
        Scale accumulated gradients by 1/batch_size to get average gradient.

        CRITICAL FIX: Gradients are summed across batch, so we must divide by batch_size
        to get the proper average gradient for SGD.
        """
        scale = 1.0 / float(batch_size)

        # Scale on GPU by multiplying each gradient by scale factor
        # For simplicity, download-scale-upload (TODO: optimize with GPU scale kernel)

        # Conv1
        conv1_w_grad = np.empty_like(self.model.conv1_weight)
        conv1_b_grad = np.empty_like(self.model.conv1_bias)
        loader.memcpy_dtoh(conv1_w_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_conv1_w, conv1_w_grad.nbytes)
        loader.memcpy_dtoh(conv1_b_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_conv1_b, conv1_b_grad.nbytes)
        conv1_w_grad *= scale
        conv1_b_grad *= scale
        loader.memcpy_htod(self.d_grad_conv1_w, conv1_w_grad.ctypes.data_as(ctypes.c_void_p), conv1_w_grad.nbytes)
        loader.memcpy_htod(self.d_grad_conv1_b, conv1_b_grad.ctypes.data_as(ctypes.c_void_p), conv1_b_grad.nbytes)

        # BN1
        bn1_gamma_grad = np.empty_like(self.model.bn1_gamma)
        bn1_beta_grad = np.empty_like(self.model.bn1_beta)
        loader.memcpy_dtoh(bn1_gamma_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_bn1_gamma, bn1_gamma_grad.nbytes)
        loader.memcpy_dtoh(bn1_beta_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_bn1_beta, bn1_beta_grad.nbytes)
        bn1_gamma_grad *= scale
        bn1_beta_grad *= scale
        loader.memcpy_htod(self.d_grad_bn1_gamma, bn1_gamma_grad.ctypes.data_as(ctypes.c_void_p), bn1_gamma_grad.nbytes)
        loader.memcpy_htod(self.d_grad_bn1_beta, bn1_beta_grad.ctypes.data_as(ctypes.c_void_p), bn1_beta_grad.nbytes)

        # Conv2
        conv2_w_grad = np.empty_like(self.model.conv2_weight)
        conv2_b_grad = np.empty_like(self.model.conv2_bias)
        loader.memcpy_dtoh(conv2_w_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_conv2_w, conv2_w_grad.nbytes)
        loader.memcpy_dtoh(conv2_b_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_conv2_b, conv2_b_grad.nbytes)
        conv2_w_grad *= scale
        conv2_b_grad *= scale
        loader.memcpy_htod(self.d_grad_conv2_w, conv2_w_grad.ctypes.data_as(ctypes.c_void_p), conv2_w_grad.nbytes)
        loader.memcpy_htod(self.d_grad_conv2_b, conv2_b_grad.ctypes.data_as(ctypes.c_void_p), conv2_b_grad.nbytes)

        # BN2
        bn2_gamma_grad = np.empty_like(self.model.bn2_gamma)
        bn2_beta_grad = np.empty_like(self.model.bn2_beta)
        loader.memcpy_dtoh(bn2_gamma_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_bn2_gamma, bn2_gamma_grad.nbytes)
        loader.memcpy_dtoh(bn2_beta_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_bn2_beta, bn2_beta_grad.nbytes)
        bn2_gamma_grad *= scale
        bn2_beta_grad *= scale
        loader.memcpy_htod(self.d_grad_bn2_gamma, bn2_gamma_grad.ctypes.data_as(ctypes.c_void_p), bn2_gamma_grad.nbytes)
        loader.memcpy_htod(self.d_grad_bn2_beta, bn2_beta_grad.ctypes.data_as(ctypes.c_void_p), bn2_beta_grad.nbytes)

        # Conv3
        conv3_w_grad = np.empty_like(self.model.conv3_weight)
        conv3_b_grad = np.empty_like(self.model.conv3_bias)
        loader.memcpy_dtoh(conv3_w_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_conv3_w, conv3_w_grad.nbytes)
        loader.memcpy_dtoh(conv3_b_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_conv3_b, conv3_b_grad.nbytes)
        conv3_w_grad *= scale
        conv3_b_grad *= scale
        loader.memcpy_htod(self.d_grad_conv3_w, conv3_w_grad.ctypes.data_as(ctypes.c_void_p), conv3_w_grad.nbytes)
        loader.memcpy_htod(self.d_grad_conv3_b, conv3_b_grad.ctypes.data_as(ctypes.c_void_p), conv3_b_grad.nbytes)

        # BN3
        bn3_gamma_grad = np.empty_like(self.model.bn3_gamma)
        bn3_beta_grad = np.empty_like(self.model.bn3_beta)
        loader.memcpy_dtoh(bn3_gamma_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_bn3_gamma, bn3_gamma_grad.nbytes)
        loader.memcpy_dtoh(bn3_beta_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_bn3_beta, bn3_beta_grad.nbytes)
        bn3_gamma_grad *= scale
        bn3_beta_grad *= scale
        loader.memcpy_htod(self.d_grad_bn3_gamma, bn3_gamma_grad.ctypes.data_as(ctypes.c_void_p), bn3_gamma_grad.nbytes)
        loader.memcpy_htod(self.d_grad_bn3_beta, bn3_beta_grad.ctypes.data_as(ctypes.c_void_p), bn3_beta_grad.nbytes)

        # FC
        fc_weight_grad = np.empty_like(self.fc_weight)
        fc_bias_grad = np.empty_like(self.fc_bias)
        loader.memcpy_dtoh(fc_weight_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_fc_weight, fc_weight_grad.nbytes)
        loader.memcpy_dtoh(fc_bias_grad.ctypes.data_as(ctypes.c_void_p), self.d_grad_fc_bias, fc_bias_grad.nbytes)
        fc_weight_grad *= scale
        fc_bias_grad *= scale
        loader.memcpy_htod(self.d_grad_fc_weight, fc_weight_grad.ctypes.data_as(ctypes.c_void_p), fc_weight_grad.nbytes)
        loader.memcpy_htod(self.d_grad_fc_bias, fc_bias_grad.ctypes.data_as(ctypes.c_void_p), fc_bias_grad.nbytes)

    def _log_gradient_norms(self):
        """Monitor gradient norms for diagnostics and debugging."""
        layer_specs = [
            ("conv1_w", self.d_grad_conv1_w, self.model.conv1_weight.shape),
            ("conv1_b", self.d_grad_conv1_b, self.model.conv1_bias.shape),
            ("bn1_gamma", self.d_grad_bn1_gamma, self.model.bn1_gamma.shape),
            ("bn1_beta", self.d_grad_bn1_beta, self.model.bn1_beta.shape),
            ("conv2_w", self.d_grad_conv2_w, self.model.conv2_weight.shape),
            ("conv2_b", self.d_grad_conv2_b, self.model.conv2_bias.shape),
            ("bn2_gamma", self.d_grad_bn2_gamma, self.model.bn2_gamma.shape),
            ("bn2_beta", self.d_grad_bn2_beta, self.model.bn2_beta.shape),
            ("conv3_w", self.d_grad_conv3_w, self.model.conv3_weight.shape),
            ("conv3_b", self.d_grad_conv3_b, self.model.conv3_bias.shape),
            ("bn3_gamma", self.d_grad_bn3_gamma, self.model.bn3_gamma.shape),
            ("bn3_beta", self.d_grad_bn3_beta, self.model.bn3_beta.shape),
            ("fc_weight", self.d_grad_fc_weight, self.fc_weight.shape),
            ("fc_bias", self.d_grad_fc_bias, self.fc_bias.shape),
        ]

        stats = []
        for name, ptr, shape in layer_specs:
            grad = np.empty(shape, dtype=np.float32)
            loader.memcpy_dtoh(grad.ctypes.data_as(ctypes.c_void_p), ptr, grad.nbytes)
            norm = float(np.linalg.norm(grad))
            stats.append(f"{name}:{norm:7.3e}")

        print("[Gradients]", " ".join(stats))

    def _normalize_gradient_norms(self, max_norm: float = 100.0):
        """
        Apply simple layer-wise gradient normalization to prevent early-layer explosions.

        Any gradient buffer with L2 norm greater than max_norm is scaled down to max_norm.
        """
        layer_specs = [
            (self.d_grad_conv1_w, self.model.conv1_weight.shape),
            (self.d_grad_conv1_b, self.model.conv1_bias.shape),
            (self.d_grad_bn1_gamma, self.model.bn1_gamma.shape),
            (self.d_grad_bn1_beta, self.model.bn1_beta.shape),
            (self.d_grad_conv2_w, self.model.conv2_weight.shape),
            (self.d_grad_conv2_b, self.model.conv2_bias.shape),
            (self.d_grad_bn2_gamma, self.model.bn2_gamma.shape),
            (self.d_grad_bn2_beta, self.model.bn2_beta.shape),
            (self.d_grad_conv3_w, self.model.conv3_weight.shape),
            (self.d_grad_conv3_b, self.model.conv3_bias.shape),
            (self.d_grad_bn3_gamma, self.model.bn3_gamma.shape),
            (self.d_grad_bn3_beta, self.model.bn3_beta.shape),
            (self.d_grad_fc_weight, self.fc_weight.shape),
            (self.d_grad_fc_bias, self.fc_bias.shape),
        ]

        for d_ptr, shape in layer_specs:
            grad = np.empty(shape, dtype=np.float32)
            loader.memcpy_dtoh(grad.ctypes.data_as(ctypes.c_void_p), d_ptr, grad.nbytes)
            norm = float(np.linalg.norm(grad))
            if norm > max_norm and norm > 0.0:
                scale = max_norm / norm
                grad *= scale
                loader.memcpy_htod(d_ptr, grad.ctypes.data_as(ctypes.c_void_p), grad.nbytes)

    def _update_all_weights(self):
        """
        Apply SGD with momentum to ALL model weights using accumulated gradients.

        CRITICAL FIX: This is called ONCE per batch, after all gradients are accumulated and scaled.
        This ensures each weight sees the average gradient across the batch, not per-sample updates.
        """
        # Update FC weights
        self.gpu_backward.sgd_momentum_update(
            self.d_fc_weight, self.d_grad_fc_weight, self.d_vel_fc_weight,
            self.learning_rate, self.momentum, self.fc_weight.size
        )
        self.gpu_backward.sgd_momentum_update(
            self.d_fc_bias, self.d_grad_fc_bias, self.d_vel_fc_bias,
            self.learning_rate, self.momentum, self.fc_bias.size
        )

        if self.fc_only:
            loader.memcpy_dtoh(self.fc_weight.ctypes.data_as(ctypes.c_void_p), self.d_fc_weight, self.fc_weight.nbytes)
            loader.memcpy_dtoh(self.fc_bias.ctypes.data_as(ctypes.c_void_p), self.d_fc_bias, self.fc_bias.nbytes)
            return

        # Update Conv3 weights
        d_conv3_w = loader.gpu_malloc(self.model.conv3_weight.nbytes)
        d_conv3_b = loader.gpu_malloc(self.model.conv3_bias.nbytes)
        loader.memcpy_htod(d_conv3_w, self.model.conv3_weight.ctypes.data_as(ctypes.c_void_p), self.model.conv3_weight.nbytes)
        loader.memcpy_htod(d_conv3_b, self.model.conv3_bias.ctypes.data_as(ctypes.c_void_p), self.model.conv3_bias.nbytes)

        self.gpu_backward.sgd_momentum_update(
            d_conv3_w, self.d_grad_conv3_w, self.d_vel_conv3_w,
            self.learning_rate, self.momentum, self.model.conv3_weight.size
        )
        self.gpu_backward.sgd_momentum_update(
            d_conv3_b, self.d_grad_conv3_b, self.d_vel_conv3_b,
            self.learning_rate, self.momentum, self.model.conv3_bias.size
        )

        loader.memcpy_dtoh(self.model.conv3_weight.ctypes.data_as(ctypes.c_void_p), d_conv3_w, self.model.conv3_weight.nbytes)
        loader.memcpy_dtoh(self.model.conv3_bias.ctypes.data_as(ctypes.c_void_p), d_conv3_b, self.model.conv3_bias.nbytes)
        loader.gpu_free(d_conv3_w)
        loader.gpu_free(d_conv3_b)

        # Update BN3 gamma/beta
        d_bn3_gamma = loader.gpu_malloc(self.model.bn3_gamma.nbytes)
        d_bn3_beta = loader.gpu_malloc(self.model.bn3_beta.nbytes)
        loader.memcpy_htod(d_bn3_gamma, self.model.bn3_gamma.ctypes.data_as(ctypes.c_void_p), self.model.bn3_gamma.nbytes)
        loader.memcpy_htod(d_bn3_beta, self.model.bn3_beta.ctypes.data_as(ctypes.c_void_p), self.model.bn3_beta.nbytes)

        self.gpu_backward.sgd_momentum_update(
            d_bn3_gamma, self.d_grad_bn3_gamma, self.d_vel_bn3_gamma,
            self.learning_rate, self.momentum, self.model.bn3_gamma.size
        )
        self.gpu_backward.sgd_momentum_update(
            d_bn3_beta, self.d_grad_bn3_beta, self.d_vel_bn3_beta,
            self.learning_rate, self.momentum, self.model.bn3_beta.size
        )

        loader.memcpy_dtoh(self.model.bn3_gamma.ctypes.data_as(ctypes.c_void_p), d_bn3_gamma, self.model.bn3_gamma.nbytes)
        loader.memcpy_dtoh(self.model.bn3_beta.ctypes.data_as(ctypes.c_void_p), d_bn3_beta, self.model.bn3_beta.nbytes)
        loader.gpu_free(d_bn3_gamma)
        loader.gpu_free(d_bn3_beta)

        # Update Conv2 weights
        d_conv2_w = loader.gpu_malloc(self.model.conv2_weight.nbytes)
        d_conv2_b = loader.gpu_malloc(self.model.conv2_bias.nbytes)
        loader.memcpy_htod(d_conv2_w, self.model.conv2_weight.ctypes.data_as(ctypes.c_void_p), self.model.conv2_weight.nbytes)
        loader.memcpy_htod(d_conv2_b, self.model.conv2_bias.ctypes.data_as(ctypes.c_void_p), self.model.conv2_bias.nbytes)

        self.gpu_backward.sgd_momentum_update(
            d_conv2_w, self.d_grad_conv2_w, self.d_vel_conv2_w,
            self.learning_rate, self.momentum, self.model.conv2_weight.size
        )
        self.gpu_backward.sgd_momentum_update(
            d_conv2_b, self.d_grad_conv2_b, self.d_vel_conv2_b,
            self.learning_rate, self.momentum, self.model.conv2_bias.size
        )

        loader.memcpy_dtoh(self.model.conv2_weight.ctypes.data_as(ctypes.c_void_p), d_conv2_w, self.model.conv2_weight.nbytes)
        loader.memcpy_dtoh(self.model.conv2_bias.ctypes.data_as(ctypes.c_void_p), d_conv2_b, self.model.conv2_bias.nbytes)
        loader.gpu_free(d_conv2_w)
        loader.gpu_free(d_conv2_b)

        # Update BN2 gamma/beta
        d_bn2_gamma = loader.gpu_malloc(self.model.bn2_gamma.nbytes)
        d_bn2_beta = loader.gpu_malloc(self.model.bn2_beta.nbytes)
        loader.memcpy_htod(d_bn2_gamma, self.model.bn2_gamma.ctypes.data_as(ctypes.c_void_p), self.model.bn2_gamma.nbytes)
        loader.memcpy_htod(d_bn2_beta, self.model.bn2_beta.ctypes.data_as(ctypes.c_void_p), self.model.bn2_beta.nbytes)

        self.gpu_backward.sgd_momentum_update(
            d_bn2_gamma, self.d_grad_bn2_gamma, self.d_vel_bn2_gamma,
            self.learning_rate, self.momentum, self.model.bn2_gamma.size
        )
        self.gpu_backward.sgd_momentum_update(
            d_bn2_beta, self.d_grad_bn2_beta, self.d_vel_bn2_beta,
            self.learning_rate, self.momentum, self.model.bn2_beta.size
        )

        loader.memcpy_dtoh(self.model.bn2_gamma.ctypes.data_as(ctypes.c_void_p), d_bn2_gamma, self.model.bn2_gamma.nbytes)
        loader.memcpy_dtoh(self.model.bn2_beta.ctypes.data_as(ctypes.c_void_p), d_bn2_beta, self.model.bn2_beta.nbytes)
        loader.gpu_free(d_bn2_gamma)
        loader.gpu_free(d_bn2_beta)

        # Update Conv1 weights
        d_conv1_w = loader.gpu_malloc(self.model.conv1_weight.nbytes)
        d_conv1_b = loader.gpu_malloc(self.model.conv1_bias.nbytes)
        loader.memcpy_htod(d_conv1_w, self.model.conv1_weight.ctypes.data_as(ctypes.c_void_p), self.model.conv1_weight.nbytes)
        loader.memcpy_htod(d_conv1_b, self.model.conv1_bias.ctypes.data_as(ctypes.c_void_p), self.model.conv1_bias.nbytes)

        self.gpu_backward.sgd_momentum_update(
            d_conv1_w, self.d_grad_conv1_w, self.d_vel_conv1_w,
            self.learning_rate, self.momentum, self.model.conv1_weight.size
        )
        self.gpu_backward.sgd_momentum_update(
            d_conv1_b, self.d_grad_conv1_b, self.d_vel_conv1_b,
            self.learning_rate, self.momentum, self.model.conv1_bias.size
        )

        loader.memcpy_dtoh(self.model.conv1_weight.ctypes.data_as(ctypes.c_void_p), d_conv1_w, self.model.conv1_weight.nbytes)
        loader.memcpy_dtoh(self.model.conv1_bias.ctypes.data_as(ctypes.c_void_p), d_conv1_b, self.model.conv1_bias.nbytes)
        loader.gpu_free(d_conv1_w)
        loader.gpu_free(d_conv1_b)

        # Update BN1 gamma/beta
        d_bn1_gamma = loader.gpu_malloc(self.model.bn1_gamma.nbytes)
        d_bn1_beta = loader.gpu_malloc(self.model.bn1_beta.nbytes)
        loader.memcpy_htod(d_bn1_gamma, self.model.bn1_gamma.ctypes.data_as(ctypes.c_void_p), self.model.bn1_gamma.nbytes)
        loader.memcpy_htod(d_bn1_beta, self.model.bn1_beta.ctypes.data_as(ctypes.c_void_p), self.model.bn1_beta.nbytes)

        self.gpu_backward.sgd_momentum_update(
            d_bn1_gamma, self.d_grad_bn1_gamma, self.d_vel_bn1_gamma,
            self.learning_rate, self.momentum, self.model.bn1_gamma.size
        )
        self.gpu_backward.sgd_momentum_update(
            d_bn1_beta, self.d_grad_bn1_beta, self.d_vel_bn1_beta,
            self.learning_rate, self.momentum, self.model.bn1_beta.size
        )

        loader.memcpy_dtoh(self.model.bn1_gamma.ctypes.data_as(ctypes.c_void_p), d_bn1_gamma, self.model.bn1_gamma.nbytes)
        loader.memcpy_dtoh(self.model.bn1_beta.ctypes.data_as(ctypes.c_void_p), d_bn1_beta, self.model.bn1_beta.nbytes)
        loader.gpu_free(d_bn1_gamma)
        loader.gpu_free(d_bn1_beta)

        # Download updated FC weights
        loader.memcpy_dtoh(self.fc_weight.ctypes.data_as(ctypes.c_void_p), self.d_fc_weight, self.fc_weight.nbytes)
        loader.memcpy_dtoh(self.fc_bias.ctypes.data_as(ctypes.c_void_p), self.d_fc_bias, self.fc_bias.nbytes)

    def train_batch(
        self,
        images: List[np.ndarray],
        labels: List[int]
    ) -> Tuple[float, float]:
        """
        Train on a batch of images using proper mini-batch SGD.

        CRITICAL FIX: Implements correct gradient accumulation:
        1. Zero gradients at batch start
        2. Accumulate gradients for each sample
        3. Scale by 1/batch_size
        4. Update weights ONCE per batch

        Args:
            images: List of images [H, W, 3], normalized to [0, 1]
            labels: List of class labels

        Returns:
            avg_loss: Average loss for batch
            accuracy: Classification accuracy
        """
        batch_size = len(images)
        total_loss = 0.0
        correct = 0

        # Step 1: Zero all gradients at the start of batch
        self._zero_gradients()

        # Step 2: Accumulate gradients for each sample in the batch
        for img, label in zip(images, labels):
            # Forward pass
            logits, probs, cache = self.forward(img)

            # Backward pass - accumulate gradients (NO weight updates)
            loss = self.accumulate_gradients(img, label, cache)

            total_loss += loss

            # Check prediction
            pred = int(np.argmax(probs))
            if pred == label:
                correct += 1

        # Step 3: Scale accumulated gradients by 1/batch_size
        self._scale_gradients(batch_size)

        if self.normalize_gradients:
            self._normalize_gradient_norms()

        self._batch_counter += 1
        if self._batch_counter % self.gradient_log_interval == 0:
            self._log_gradient_norms()

        # Step 4: Update ALL weights ONCE using scaled gradients
        self._update_all_weights()

        avg_loss = total_loss / batch_size
        accuracy = correct / batch_size

        return avg_loss, accuracy
