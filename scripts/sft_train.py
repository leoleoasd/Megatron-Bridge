#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SFT Training Script for Megatron Bridge

This script performs Supervised Fine-Tuning (SFT) using Megatron Bridge.
It loads a HuggingFace checkpoint, trains on JSONL data, and saves checkpoints.

Usage:
    Single node (8 GPUs):
        torchrun --nproc_per_node=8 scripts/sft_train.py \
            --model Qwen/Qwen2.5-7B-Instruct \
            --data-path /path/to/training.jsonl \
            --output-dir ./checkpoints

    Multi-node:
        torchrun --nnodes=2 --nproc_per_node=8 \
            --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            scripts/sft_train.py \
            --model Qwen/Qwen2.5-7B-Instruct \
            --data-path /path/to/training.jsonl \
            --tensor-parallel-size 4 \
            --pipeline-parallel-size 2

Data Format:
    The data file should be a JSONL file with chat-format messages:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Or standard SFT format:
    {"input": "...", "output": "..."}
"""

import argparse
import logging
import os
from dataclasses import fields
from pathlib import Path

import torch
from tqdm import tqdm

# Configure logging with timestamp
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("megatron.core.distributed.param_and_grad_buffer").setLevel(logging.WARNING)

# Enable debug logging for checkpointing
logging.getLogger("megatron.bridge.training.checkpointing").setLevel(logging.DEBUG)
logging.getLogger("megatron.core.dist_checkpointing").setLevel(logging.DEBUG)

from megatron.bridge import AutoBridge
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    FinetuningDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory


class TqdmProgressCallback(Callback):
    """Callback to display tqdm progress bar during training."""

    def __init__(self, total_iters: int, log_interval: int = 1):
        """Initialize tqdm progress callback.

        Args:
            total_iters: Total number of training iterations
            log_interval: How often to update the progress bar
        """
        self.total_iters = total_iters
        self.log_interval = log_interval
        self.pbar = None
        self.current_loss = None

    def on_train_start(self, context: CallbackContext) -> None:
        """Initialize progress bar at training start."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            start_iter = context.state.train_state.step
            self.pbar = tqdm(
                total=self.total_iters,
                initial=start_iter,
                desc="Training",
                unit="iter",
                dynamic_ncols=True,
            )

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Update progress bar after each training step."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and self.pbar is not None:
            # Update progress bar
            self.pbar.update(1)

            # Update loss display if available
            if context.loss_dict:
                loss_val = context.loss_dict.get("lm loss", None)
                if loss_val is not None:
                    if hasattr(loss_val, "item"):
                        loss_val = loss_val.item()
                    self.current_loss = loss_val
                    self.pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    def on_train_end(self, context: CallbackContext) -> None:
        """Close progress bar at training end."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and self.pbar is not None:
            self.pbar.close()



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SFT Training with Megatron Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading HuggingFace model",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training JSONL file or directory containing training.jsonl",
    )
    data_group.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: use model's max_position_embeddings)",
    )
    data_group.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Path to a Jinja2 chat template file to override the tokenizer's default chat template",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (mutually exclusive with --train-iters)",
    )
    train_group.add_argument(
        "--train-iters",
        type=int,
        default=1000,
        help="Number of training iterations (default: 1000)",
    )
    train_group.add_argument(
        "--global-batch-size",
        type=int,
        default=128,
        help="Global batch size (default: 128)",
    )
    train_group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size per GPU (default: 1)",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    train_group.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate (default: 0.0)",
    )
    train_group.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=50,
        help="Learning rate warmup iterations (default: 50)",
    )

    # Parallelism and model arguments - auto-generated from TransformerConfig
    # TransformerConfig inherits from ModelParallelConfig, so includes all parallelism settings
    # Uses ArgumentGroupFactory to automatically create args from the dataclass
    # Exclude list copied from _add_network_size_args in megatron/training/arguments.py
    transformer_exclude = [
        # cannot provide callables over CLI
        "timers",
        "finalize_model_grads_func",
        "grad_scale_func",
        "no_sync_func",
        "grad_sync_func",
        "param_sync_func",
        "_cpu_offloading_context",
        "init_method",
        "output_layer_init_method",
        "embedding_init_method",
        "activation_func",
        # types affect docstring
        "pipeline_model_parallel_layout",
        "window_size",
        "window_attn_skip_freq",
        "no_rope_freq",
        "moe_layer_freq",
        "linear_attention_freq",
        "moe_router_load_balancing_type",
        "moe_aux_loss_coeff",
        "cp_comm_type",
        "cuda_graph_scope",
        # no CLI argument exists for these
        "virtual_pipeline_model_parallel_size",
        "params_dtype",
        "enable_autocast",
        "autocast_dtype",
        "num_microbatches_with_partial_activation_checkpoints",
        "tp_comm_overlap_disable_qkv",
        "tp_comm_overlap_disable_fc1",
        "pipeline_dtype",
        "variable_seq_lengths",
        "batch_p2p_comm",
        "batch_p2p_sync",
        "deallocate_pipeline_outputs",
        "cpu_offloading",
        "cpu_offloading_activations",
        "cpu_offloading_weights",
        "cpu_offloading_double_buffering",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
        "softmax_scale",
        "gated_linear_unit",
        "bias_activation_fusion",
        "activation_func_fp8_input_store",
        "test_mode",
        "memory_efficient_layer_norm",
        "fused_single_qkv_rope",
        "fp8_dot_product_attention",
        "fp8_multi_head_attention",
        "tp_only_amax_red",
        "use_kitchen",
        "moe_token_dropping",
        "cuda_graph_use_single_mempool",
        "cuda_graph_retain_backward_graph",
        "disable_parameter_transpose_cache",
        "inference_sampling_seed",
        "use_inference_optimized_layers",
        "heterogeneous_block_specs",
        "hetereogenous_dist_checkpoint",
        "quant_recipe",
        # deprecated and no CLI arg exists
        "tp_comm_atomic_ag",
        "tp_comm_atomic_rs",
        "moe_router_topk_limited_devices",
        # already generated by another config
        "inference_rng_tracker",
        "use_te_rng_tracker",
        "log_max_attention_logit",
        "barrier_with_L1_time",
        # args uses same var with a different name
        "num_moe_experts",
        "fp8_param",
        # incompatible defaults in dataclass
        "gradient_accumulation_fusion",
        "overlap_p2p_comm",
        "attention_softmax_in_fp32",
        "masked_softmax_fusion",
        "persist_layer_norm",
        "bias_dropout_fusion",
        "apply_rope_fusion",
    ]
    transformer_factory = ArgumentGroupFactory(TransformerConfig, exclude=transformer_exclude)
    transformer_factory.build_group(parser, "Model & Parallelism")

    # Checkpoint arguments
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--output-dir",
        type=str,
        default="./nemo_experiments/sft",
        help="Output directory for checkpoints and logs",
    )
    ckpt_group.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save checkpoint every N iterations, or every N epochs when --epochs is used (default: 100)",
    )
    ckpt_group.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint directory",
    )

    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (enables W&B logging)",
    )
    log_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    log_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name",
    )

    # Validation arguments
    val_group = parser.add_argument_group("Validation")
    val_group.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate every N iterations (default: 100)",
    )
    val_group.add_argument(
        "--eval-iters",
        type=int,
        default=10,
        help="Number of evaluation iterations (default: 10)",
    )

    # Advanced arguments
    advanced_group = parser.add_argument_group("Advanced")
    advanced_group.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Random seed (default: 5678)",
    )
    advanced_group.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar",
    )

    return parser.parse_args()


def create_sft_config(args: argparse.Namespace) -> ConfigContainer:
    """Create SFT configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        ConfigContainer with SFT configuration
    """
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tb_logs"

    # Create model provider from HuggingFace model
    bridge = AutoBridge.from_hf_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)

    # Copy all ModelParallelConfig fields from args to model_provider
    # This includes parallelism settings, precision, communication overlaps, etc.
    for field in fields(ModelParallelConfig):
        field_name = field.name
        if hasattr(args, field_name):
            arg_value = getattr(args, field_name)
            if arg_value is not None and hasattr(model_provider, field_name):
                setattr(model_provider, field_name, arg_value)

    # Set pipeline dtype if PP > 1
    if args.pipeline_model_parallel_size > 1:
        model_provider.pipeline_dtype = torch.bfloat16

    model_provider.calculate_per_token_loss = True

    # Configure recomputation if specified (from TransformerConfig)
    if args.recompute_granularity:
        model_provider.recompute_granularity = args.recompute_granularity
    if args.recompute_method:
        model_provider.recompute_method = args.recompute_method
    if args.recompute_num_layers is not None:
        model_provider.recompute_num_layers = args.recompute_num_layers

    # Determine data path
    data_path = Path(args.data_path)
    if data_path.is_file():
        # Single file provided - use parent directory as dataset root
        dataset_root = data_path.parent
    else:
        # Directory provided
        dataset_root = data_path

    dataset_kwargs = {}
    # Always use chat format with HF tokenizer chat template
    dataset_kwargs["chat"] = True
    dataset_kwargs["use_hf_tokenizer_chat_template"] = True

    # Load custom chat template if provided
    chat_template_content = None
    if args.chat_template:
        chat_template_path = Path(args.chat_template)
        if not chat_template_path.exists():
            raise FileNotFoundError(f"Chat template file not found: {args.chat_template}")
        with open(chat_template_path, "r") as f:
            chat_template_content = f.read()

    # Create optimizer and scheduler config
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=args.lr_warmup_iters,
        lr_decay_iters=args.train_iters,
        max_lr=args.lr,
        min_lr=args.min_lr,
        adam_beta2=0.95,
    )

    # Configure wandb in logger if enabled
    wandb_project = args.wandb_project
    wandb_exp_name = args.wandb_run_name
    wandb_entity = args.wandb_entity

    # Determine sequence length: use provided value or model's max_position_embeddings
    seq_length = args.seq_length if args.seq_length is not None else model_provider.seq_length

    # Update model provider's seq_length to match dataset (required for config validation)
    model_provider.seq_length = seq_length

    # Create the config container
    cfg = ConfigContainer(
        model=model_provider,
        train=TrainingConfig(
            train_iters=args.train_iters,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
        ),
        validation=ValidationConfig(
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        dataset=FinetuningDatasetConfig(
            dataset_root=str(dataset_root),
            seq_length=seq_length,
            seed=args.seed,
            memmap_workers=20,
            dataset_kwargs=dataset_kwargs,
            do_validation=False,
            do_test=False,
            dataloader_type="batch",
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir=str(tensorboard_dir),
            log_timers_to_tensorboard=True,
            wandb_project=wandb_project,
            wandb_exp_name=wandb_exp_name,
            wandb_entity=wandb_entity,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=args.model,
            chat_template=chat_template_content,
        ),
        checkpoint=CheckpointConfig(
            save_interval=args.save_interval,
            save=str(checkpoint_dir),
            load=str(checkpoint_dir) if args.resume_from is None else args.resume_from,
            pretrained_checkpoint=None,  # We're loading weights directly via AutoBridge
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=True,
            use_persistent_ckpt_worker=True,
            ckpt_assume_constant_structure=True,  # Cache checkpoint structure for faster saves
            save_optim=False,  # Don't save optimizer state (much smaller checkpoints)
            save_rng=False,  # Don't save RNG state
        ),
        mixed_precision=bf16_mixed() if args.bf16 else None,
        rng=RNGConfig(seed=args.seed),
    )

    # Configure DDP settings
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True

    return cfg


def count_jsonl_samples(data_path: Path) -> int:
    """Count the number of samples in a JSONL file.

    Args:
        data_path: Path to the JSONL file or directory containing training.jsonl

    Returns:
        Number of samples (lines) in the file
    """
    if data_path.is_dir():
        data_path = data_path / "training.jsonl"

    count = 0
    with open(data_path, "r") as f:
        for _ in f:
            count += 1
    return count


def calculate_train_iters(num_samples: int, global_batch_size: int, epochs: int) -> int:
    """Calculate the number of training iterations for a given number of epochs.

    Args:
        num_samples: Total number of samples in the dataset
        global_batch_size: Global batch size
        epochs: Number of epochs to train

    Returns:
        Number of training iterations
    """
    steps_per_epoch = num_samples // global_batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    return steps_per_epoch * epochs


def main() -> None:
    """Main entry point for SFT training."""
    args = parse_args()

    # Handle epochs vs train_iters
    if args.epochs is not None:
        data_path = Path(args.data_path)
        num_samples = count_jsonl_samples(data_path)
        steps_per_epoch = num_samples // args.global_batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        args.train_iters = calculate_train_iters(num_samples, args.global_batch_size, args.epochs)
        # When training by epochs, interpret --save-interval as every N epochs
        args.save_interval = args.save_interval * steps_per_epoch
        print(f"Dataset size: {num_samples} samples")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Training for {args.epochs} epochs = {args.train_iters} iterations")
        print(f"Save interval: every {args.save_interval // steps_per_epoch} epoch(s) = {args.save_interval} iterations")

    # Validate parallelism settings
    if args.sequence_parallel and args.tensor_model_parallel_size <= 1:
        print("Warning: --sequence-parallel requires --tensor-model-parallel-size > 1. Disabling sequence parallelism.")
        args.sequence_parallel = False

    # Set CUDA_DEVICE_MAX_CONNECTIONS for sequence parallelism speedup
    if args.sequence_parallel:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create configuration
    config = create_sft_config(args)

    # Create callbacks list
    callbacks = []

    # Add tqdm progress callback (rank 0 only will actually display)
    if not args.no_tqdm:
        callbacks.append(
            TqdmProgressCallback(
                total_iters=args.train_iters,
                log_interval=1,
            )
        )

    # Print configuration summary
    print("=" * 60)
    print("SFT Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data path: {args.data_path}")
    print(f"Sequence length: {config.dataset.seq_length}")
    print(f"Global batch size: {args.global_batch_size}")
    print(f"Micro batch size: {args.micro_batch_size}")
    if args.epochs is not None:
        print(f"Epochs: {args.epochs}")
    print(f"Training iterations: {args.train_iters}")
    print(f"Learning rate: {args.lr}")
    print(f"Tensor model parallel size: {args.tensor_model_parallel_size}")
    print(f"Pipeline model parallel size: {args.pipeline_model_parallel_size}")
    print(f"Context parallel size: {args.context_parallel_size}")
    print(f"Expert model parallel size: {args.expert_model_parallel_size}")
    print(f"Expert tensor parallel size: {args.expert_tensor_parallel_size or args.tensor_model_parallel_size}")
    print(f"Sequence parallel: {args.sequence_parallel}")
    print(f"Output directory: {args.output_dir}")
    if args.wandb_project:
        print(f"W&B Project: {args.wandb_project}")
        print(f"W&B Run Name: {args.wandb_run_name or 'auto'}")
    print("=" * 60)

    # Start training
    finetune(config=config, forward_step_func=forward_step, callbacks=callbacks)


if __name__ == "__main__":
    main()
