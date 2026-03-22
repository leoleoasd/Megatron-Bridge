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
import os
from pathlib import Path

import torch
from tqdm import tqdm

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


class WandbLoggingCallback(Callback):
    """Callback to log metrics to Weights & Biases."""

    def __init__(self, project: str, run_name: str = None, config: dict = None):
        """Initialize wandb logging callback.

        Args:
            project: W&B project name
            run_name: Optional run name
            config: Optional config dict to log
        """
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.wandb_run = None

    def on_train_start(self, context: CallbackContext) -> None:
        """Initialize wandb at training start (rank 0 only)."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=self.project,
                    name=self.run_name,
                    config=self.config,
                    resume="allow",
                )
                print(f"W&B run initialized: {wandb.run.url}")
            except Exception as e:
                print(f"Failed to initialize W&B: {e}")
                self.wandb_run = None

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Log metrics to wandb after each training step."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and self.wandb_run is not None:
            try:
                import wandb

                step = context.state.train_state.step
                metrics = {"iteration": step}

                # Log losses
                if context.loss_dict:
                    for key, val in context.loss_dict.items():
                        if hasattr(val, "item"):
                            val = val.item()
                        # Convert key to wandb-friendly format
                        wandb_key = f"train/{key.replace(' ', '_')}"
                        metrics[wandb_key] = val

                # Log gradient norm if available
                if context.grad_norm is not None:
                    metrics["train/grad_norm"] = context.grad_norm

                # Log learning rate
                if context.scheduler is not None:
                    try:
                        lr = context.scheduler.get_lr()
                        metrics["train/learning_rate"] = lr
                    except Exception:
                        pass

                wandb.log(metrics, step=step)
            except Exception as e:
                # Don't fail training on wandb errors
                pass

    def on_train_end(self, context: CallbackContext) -> None:
        """Finish wandb run at training end."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and self.wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


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
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    data_group.add_argument(
        "--chat-format",
        action="store_true",
        default=True,
        help="Use chat format for data (messages format)",
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

    # Parallelism arguments
    parallel_group = parser.add_argument_group("Parallelism")
    parallel_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parallel_group.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size (default: 1)",
    )
    parallel_group.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Context parallel size for long sequences (default: 1)",
    )
    parallel_group.add_argument(
        "--expert-parallel-size",
        type=int,
        default=1,
        help="Expert parallel size for MoE models (default: 1)",
    )
    parallel_group.add_argument(
        "--sequence-parallel",
        action="store_true",
        default=False,
        help="Enable sequence parallelism (requires TP > 1)",
    )
    parallel_group.add_argument(
        "--virtual-pipeline-parallel-size",
        type=int,
        default=None,
        help="Virtual pipeline parallel size for interleaved scheduling",
    )

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
        help="Save checkpoint every N iterations (default: 100)",
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
        "--log-interval",
        type=int,
        default=1,
        help="Log every N iterations (default: 1)",
    )
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
        "--recompute-granularity",
        type=str,
        default=None,
        choices=["full", "selective", None],
        help="Activation recomputation granularity for memory savings",
    )
    advanced_group.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Random seed (default: 5678)",
    )
    advanced_group.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 mixed precision (default: True)",
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

    # Configure parallelism
    model_provider.tensor_model_parallel_size = args.tensor_parallel_size
    model_provider.pipeline_model_parallel_size = args.pipeline_parallel_size
    model_provider.context_parallel_size = args.context_parallel_size
    model_provider.sequence_parallel = args.sequence_parallel
    model_provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_parallel_size

    # Configure expert parallelism for MoE models
    if args.expert_parallel_size > 1:
        model_provider.expert_model_parallel_size = args.expert_parallel_size

    # Set pipeline dtype if PP > 1
    if args.pipeline_parallel_size > 1:
        model_provider.pipeline_dtype = torch.bfloat16

    # Configure sequence length
    model_provider.seq_length = args.seq_length

    # Configure recomputation if specified
    if args.recompute_granularity:
        model_provider.recompute_granularity = args.recompute_granularity

    # Determine data path
    data_path = Path(args.data_path)
    if data_path.is_file():
        # Single file provided - use parent directory as dataset root
        dataset_root = data_path.parent
    else:
        # Directory provided
        dataset_root = data_path

    dataset_kwargs = {}
    # Add chat-specific kwargs
    if args.chat_format:
        dataset_kwargs["chat"] = True
        dataset_kwargs["use_hf_tokenizer_chat_template"] = True

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
            seq_length=args.seq_length,
            seed=args.seed,
            memmap_workers=20,
            dataset_kwargs=dataset_kwargs,
            do_validation=False,
            do_test=False,
            dataloader_type="batch",
        ),
        logger=LoggerConfig(
            log_interval=args.log_interval,
            tensorboard_dir=str(tensorboard_dir),
            log_timers_to_tensorboard=True,
            wandb_project=wandb_project,
            wandb_exp_name=wandb_exp_name,
            wandb_entity=wandb_entity,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=args.model,
        ),
        checkpoint=CheckpointConfig(
            save_interval=args.save_interval,
            save=str(checkpoint_dir),
            load=str(checkpoint_dir) if args.resume_from is None else args.resume_from,
            pretrained_checkpoint=None,  # We're loading weights directly via AutoBridge
            ckpt_format="torch_dist",
            fully_parallel_save=True,
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
        print(f"Dataset size: {num_samples} samples")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Training for {args.epochs} epochs = {args.train_iters} iterations")

    # Validate parallelism settings
    if args.sequence_parallel and args.tensor_parallel_size <= 1:
        print("Warning: --sequence-parallel requires --tensor-parallel-size > 1. Disabling sequence parallelism.")
        args.sequence_parallel = False

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
                log_interval=args.log_interval,
            )
        )

    # Add wandb callback if project is specified
    if args.wandb_project:
        wandb_config = {
            "model": args.model,
            "data_path": args.data_path,
            "seq_length": args.seq_length,
            "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "epochs": args.epochs,
            "train_iters": args.train_iters,
            "learning_rate": args.lr,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "context_parallel_size": args.context_parallel_size,
            "expert_parallel_size": args.expert_parallel_size,
            "sequence_parallel": args.sequence_parallel,
        }
        callbacks.append(
            WandbLoggingCallback(
                project=args.wandb_project,
                run_name=args.wandb_run_name,
                config=wandb_config,
            )
        )

    # Print configuration summary
    print("=" * 60)
    print("SFT Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data path: {args.data_path}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Global batch size: {args.global_batch_size}")
    print(f"Micro batch size: {args.micro_batch_size}")
    if args.epochs is not None:
        print(f"Epochs: {args.epochs}")
    print(f"Training iterations: {args.train_iters}")
    print(f"Learning rate: {args.lr}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Pipeline parallel size: {args.pipeline_parallel_size}")
    print(f"Context parallel size: {args.context_parallel_size}")
    print(f"Expert parallel size: {args.expert_parallel_size}")
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
