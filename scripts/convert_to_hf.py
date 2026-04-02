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
Megatron to HuggingFace Checkpoint Conversion Script

This script converts Megatron Bridge checkpoints to HuggingFace format.
It can handle:
- A specific checkpoint iteration (e.g., /path/to/checkpoints/iter_0000700)
- A checkpoints directory containing multiple iterations (converts all)
- An experiment directory containing a checkpoints subdirectory

The tokenizer is always re-exported from the original HuggingFace model to ensure
correctness. By default, the chat template is cleared. Use --use-original-chat-template
to keep the original.

Usage examples:
  # Convert a specific checkpoint (clears chat template by default)
  python scripts/convert_to_hf.py \\
    --hf-model Qwen/Qwen3-30B-A3B-Thinking-2507 \\
    --input /tmp/instance_storage/nemo_test/checkpoints/iter_0000700 \\
    --output-dir ./hf_exports

  # Convert all checkpoints in a directory
  python scripts/convert_to_hf.py \\
    --hf-model Qwen/Qwen3-30B-A3B-Thinking-2507 \\
    --input /tmp/instance_storage/nemo_test/checkpoints \\
    --output-dir ./hf_exports

  # Convert with dtype conversion to float16
  python scripts/convert_to_hf.py \\
    --hf-model Qwen/Qwen3-30B-A3B-Thinking-2507 \\
    --input /tmp/instance_storage/nemo_test \\
    --output-dir ./hf_exports \\
    --dtype float16

  # Convert and keep original chat template
  python scripts/convert_to_hf.py \\
    --hf-model Qwen/Qwen3-30B-A3B-Thinking-2507 \\
    --input /tmp/instance_storage/nemo_test \\
    --output-dir ./hf_exports \\
    --use-original-chat-template
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml

# Patch Megatron Bridge's load_model_config to use the Bridge provider
# This is needed when the checkpoint was saved by Megatron without full provider info
import megatron.bridge.training.model_load_save as _model_load_save_module

_provider_override = {}
_original_load_model_config = _model_load_save_module.load_model_config


def _patched_load_model_config(checkpoint_path):
    """Patched load_model_config that uses Bridge provider when available."""
    model_cfg, mlm_args = _original_load_model_config(checkpoint_path)
    provider = _provider_override.get("provider")
    if provider is not None:
        from megatron.bridge.models.model_provider import ModelProviderMixin

        if not isinstance(model_cfg, ModelProviderMixin):
            logging.info(
                f"Overriding MLM TransformerConfig with Bridge provider: {type(provider).__name__}"
            )
            return provider, mlm_args
    return model_cfg, mlm_args


_model_load_save_module.load_model_config = _patched_load_model_config

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def find_checkpoints(input_path: Path) -> List[Path]:
    """
    Find all checkpoint directories from the input path.
    
    Args:
        input_path: Can be:
            - A specific iter_XXXXXXX directory
            - A checkpoints directory containing iter_* subdirectories
            - An experiment directory containing a checkpoints subdirectory
    
    Returns:
        List of checkpoint directories (iter_* paths)
    """
    checkpoints = []
    
    # Case 1: Direct iter_* directory
    if input_path.name.startswith("iter_") and input_path.is_dir():
        if (input_path / "run_config.yaml").exists() or (input_path / ".metadata").exists():
            checkpoints.append(input_path)
            return checkpoints
    
    # Case 2: Check if input has a checkpoints subdirectory
    checkpoints_dir = input_path / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        input_path = checkpoints_dir
    
    # Case 3: Directory containing iter_* subdirectories
    if input_path.is_dir():
        for item in input_path.iterdir():
            if item.is_dir() and item.name.startswith("iter_"):
                # Verify it's a valid checkpoint
                if (item / "run_config.yaml").exists() or (item / ".metadata").exists():
                    checkpoints.append(item)
    
    # Sort by iteration number
    def get_iter_num(path: Path) -> int:
        match = re.search(r"iter_(\d+)", path.name)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_iter_num)
    
    return checkpoints


def convert_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    hf_model_id: str,
    use_original_chat_template: bool = False,
    show_progress: bool = True,
    dtype: Optional[str] = None,
) -> Path:
    """
    Convert a single Megatron checkpoint to HuggingFace format.
    
    Args:
        checkpoint_path: Path to the iter_* checkpoint directory
        output_dir: Base output directory for HF exports
        hf_model_id: HuggingFace model ID for the base model
        use_original_chat_template: If True, keep original chat template from base model
        show_progress: Show progress bar during conversion
        dtype: Output dtype for weights (float16, bfloat16, float32). If None, uses checkpoint dtype.
    
    Returns:
        Path to the exported HuggingFace model directory
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.model_load_save import temporary_distributed_context
    
    # Parse dtype if specified
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    target_dtype = None
    if dtype is not None:
        if dtype.lower() not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(dtype_map.keys())}")
        target_dtype = dtype_map[dtype.lower()]
        logger.info(f"Will convert weights to dtype: {target_dtype}")
    
    # Create output path based on checkpoint iteration
    iter_name = checkpoint_path.name  # e.g., "iter_0000700"
    hf_output_path = output_dir / f"{iter_name}_hf"
    
    logger.info(f"Converting {checkpoint_path} -> {hf_output_path}")
    
    # Create bridge from the original HF model
    logger.info(f"Loading base model configuration from: {hf_model_id}")
    bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
    
    provider = bridge.to_megatron_provider(load_weights=False)
    _provider_override["provider"] = provider
    logger.info(f"Using Bridge provider: {type(provider).__name__}")
    
    logger.info("Exporting checkpoint to HuggingFace format...")
    
    dtype_overrides = {}
    if target_dtype == torch.bfloat16:
        dtype_overrides = {"bf16": True}
    elif target_dtype == torch.float16:
        dtype_overrides = {"fp16": True}
    elif target_dtype == torch.float32:
        dtype_overrides = {} # should be default
    else:
        raise ValueError(f"Unsupported dtype: {target_dtype}")
    with temporary_distributed_context(backend="gloo"):
        megatron_model = bridge.load_megatron_model(
            str(checkpoint_path),
            wrap_with_ddp=False,
            mp_overrides=dtype_overrides or None,
        )
        
        bridge.save_hf_pretrained(
            megatron_model,
            str(hf_output_path),
            show_progress=show_progress,
            strict=False,
        )
    
    logger.info("Exporting tokenizer from original HuggingFace model...")
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        
        if use_original_chat_template:
            logger.info("Keeping original chat template from base model")
        else:
            ckpt_chat_template_jinja = checkpoint_path / "tokenizer" / "chat_template.jinja"
            if ckpt_chat_template_jinja.exists():
                logger.info("Using chat template from checkpoint")
                tokenizer.chat_template = ckpt_chat_template_jinja.read_text()
            else:
                logger.info("No chat template in checkpoint, clearing")
                tokenizer.chat_template = None
        
        tokenizer.save_pretrained(hf_output_path)
        logger.info("Saved tokenizer to output directory")
    except Exception as e:
        logger.warning(f"Failed to export tokenizer: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"Successfully exported to: {hf_output_path}")
    return hf_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Megatron Bridge checkpoints to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model ID or path (used to read config and tokenizer)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input path: checkpoint directory, checkpoints folder, or experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace exports",
    )
    parser.add_argument(
        "--use-original-chat-template",
        action="store_true",
        default=False,
        help="Keep original chat template from base model [default: False, clears chat template]",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Output dtype for weights [default: bfloat16]",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    hf_model_id = args.hf_model
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Find all checkpoints
    checkpoints = find_checkpoints(input_path)
    
    if not checkpoints:
        logger.error(f"No valid checkpoints found in: {input_path}")
        logger.info("Expected to find iter_* directories with run_config.yaml or .metadata files")
        return 1
    
    logger.info(f"Found {len(checkpoints)} checkpoint(s) to convert:")
    for ckpt in checkpoints:
        logger.info(f"  - {ckpt.name}")
    
    logger.info(f"Using HuggingFace model for config: {hf_model_id}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each checkpoint
    successful = 0
    failed = 0
    
    for checkpoint_path in checkpoints:
        try:
            convert_checkpoint(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                hf_model_id=hf_model_id,
                use_original_chat_template=args.use_original_chat_template,
                show_progress=not args.no_progress,
                dtype=args.dtype,
            )
            successful += 1
        except Exception as e:
            logger.error(f"Failed to convert {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Conversion complete: {successful} successful, {failed} failed")
    logger.info(f"Output directory: {output_dir}")
    
    if successful > 0:
        logger.info("\nTo load a converted model:")
        logger.info("  from transformers import AutoModelForCausalLM, AutoTokenizer")
        logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}/iter_XXXXXXX_hf')")
        logger.info(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}/iter_XXXXXXX_hf')")
    
    # Cleanup distributed if initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
