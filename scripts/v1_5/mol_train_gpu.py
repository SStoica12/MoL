#!/usr/bin/env python3
"""
Dummy GPU memory occupier script.
This script allocates and holds GPU memory on specified GPUs with optional randomness.

Usage:
    python debug_gpu.py --gpus 6,7 --memory 10  # Occupy 10GB on GPUs 4,5,6,7
    python debug_gpu.py --gpus 6 --memory 5         # Occupy 5GB on GPU 0
    python debug_gpu.py --gpus 6,7 --memory-per-gpu 8,12  # Occupy 8GB on GPU 0 and 12GB on GPU 1
    python debug_gpu.py --gpus 6 --memory 60 --memory-variance 1.0  # Occupy 59-61GB (60±1GB)
"""

import argparse
import time
import sys
import os
import random

try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed. Please install it with: pip install torch")
    sys.exit(1)

def occupy_gpu_memory(gpu_id, memory_gb):
    """
    Occupy a specific amount of memory on a given GPU.
    
    Args:
        gpu_id: GPU device ID
        memory_gb: Amount of memory to occupy in GB
    
    Returns:
        Tensor holding the allocated memory
    """
    device = torch.device(f'cuda:{gpu_id}')

    # Calculate number of float32 elements needed for the specified memory
    # 1 GB = 1024^3 bytes, float32 = 4 bytes
    bytes_per_gb = 1024 ** 3
    bytes_per_float32 = 4
    num_elements = int((memory_gb * bytes_per_gb) / bytes_per_float32)

    print(f"Allocating {memory_gb:.2f} GB on GPU {gpu_id}...")

    # Allocate memory
    try:
        tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        # Fill with some data to ensure allocation
        tensor.fill_(1.0)

        # Get actual memory usage
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)

        print(f"GPU {gpu_id}: Allocated {allocated_mb:.2f} MB, Reserved {reserved_mb:.2f} MB")

        return tensor
    except RuntimeError as e:
        print(f"Error allocating memory on GPU {gpu_id}: {e}")
        return None
    
def main():
    parser = argparse.ArgumentParser(
        description="Occupy GPU memory on specified GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Occupy 10GB on GPUs 0,1,2,3
  python debug_gpu.py --gpus 0,1,2,3 --memory 10
  
  # Occupy 5GB on GPU 0
  python debug_gpu.py --gpus 0 --memory 5
  
  # Occupy different amounts on different GPUs
  python debug_gpu.py --gpus 0,1 --memory-per-gpu 8,12
  
  # Occupy 90% of available memory on GPUs 0,1
  python debug_gpu.py --gpus 0,1 --memory-fraction 0.9
  
  # Occupy 60GB with ±1GB randomness (59-61GB range)
  python debug_gpu.py --gpus 0 --memory 60 --memory-variance 1.0
  
  # Occupy 10GB with ±0.5GB randomness on multiple GPUs
  python debug_gpu.py --gpus 0,1,2,3 --memory 10 --memory-variance 0.5
        """
    )

    parser.add_argument(
        '--gpus',
        type=str,
        required=True,
        help='Comma-separated list of GPU IDs (e.g., 0,1,2,3)'
    )

    parser.add_argument(
        '--memory',
        type=float,
        default=72,
        help='Memory to occupy on each GPU in GB (applies to all GPUs)'
    )

    parser.add_argument(
        '--memory-per-gpu',
        type=str,
        default=None,
        help='Comma-separated list of memory amounts in GB for each GPU (e.g., 8,10,12,8)'
    )

    parser.add_argument(
        '--memory-fraction',
        type=float,
        default=None,
        help='Fraction of total GPU memory to occupy (0.0-1.0, e.g., 0.9 for 90%%)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration to hold memory in seconds (default: infinite, use Ctrl+C to stop)'
    )

    parser.add_argument(
        '--memory-variance',
        type=float,
        default=1.0,
        help='Memory variance in GB (e.g., 1.0 means +/- 1GB randomness around target, default: 1.0)'
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. No GPUs detected.")
        sys.exit(1)

    # Parse GPU IDs
    try:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    except ValueError:
        print("Error: Invalid GPU IDs. Please provide comma-separated integers.")
        sys.exit(1)

    # Validate GPU IDs
    num_gpus = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id >= num_gpus or gpu_id < 0:
            print(f"Error: GPU {gpu_id} is not available. Available GPUs: 0-{num_gpus-1}")
            sys.exit(1)

    # Determine memory amounts for each GPU
    memory_amounts = []

    if args.memory_per_gpu is not None:
        # Different memory for each GPU
        try:
            memory_amounts = [float(x.strip()) for x in args.memory_per_gpu.split(',')]
        except ValueError:
            print("Error: Invalid memory amounts. Please provide comma-separated numbers.")
            sys.exit(1)

        if len(memory_amounts) != len(gpu_ids):
            print(f"Error: Number of memory amounts ({len(memory_amounts)}) must match number of GPUs ({len(gpu_ids)})")
            sys.exit(1)

    elif args.memory is not None:
        # Same memory for all GPUs
        memory_amounts = [args.memory] * len(gpu_ids)

    elif args.memory_fraction is not None:
        # Fraction of available memory
        if args.memory_fraction <= 0 or args.memory_fraction > 1.0:
            print("Error: Memory fraction must be between 0 and 1.0")
            sys.exit(1)

        for gpu_id in gpu_ids:
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            total_memory_gb = total_memory / (1024 ** 3)
            target_memory_gb = total_memory_gb * args.memory_fraction
            memory_amounts.append(target_memory_gb)

    else:
        print("Error: Must specify --memory, --memory-per-gpu, or --memory-fraction")
        sys.exit(1)

    # Apply randomness to memory amounts
    if args.memory_variance > 0:
        original_amounts = memory_amounts.copy()
        memory_amounts = []
        for i, mem_gb in enumerate(original_amounts):
            # Add random variance between -memory_variance and +memory_variance
            variance = random.uniform(-args.memory_variance, args.memory_variance)
            randomized_mem = mem_gb + variance
            # Ensure we don't go below 0.1 GB
            randomized_mem = max(0.1, randomized_mem)
            memory_amounts.append(randomized_mem)
            print(f"GPU {gpu_ids[i]}: Target {mem_gb:.2f} GB -> Randomized to {randomized_mem:.2f} GB (variance: {variance:+.2f} GB)")

    # Print summary
    print("\n" + "="*60)
    print("GPU Memory Occupier")
    print("="*60)
    print(f"Total GPUs available: {num_gpus}")
    print(f"Target GPUs: {gpu_ids}")
    print("\nTarget memory allocation:")
    for gpu_id, mem_gb in zip(gpu_ids, memory_amounts):
        props = torch.cuda.get_device_properties(gpu_id)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU {gpu_id} ({props.name}): {mem_gb:.2f} GB / {total_mem_gb:.2f} GB ({mem_gb/total_mem_gb*100:.1f}%)")
    print("="*60 + "\n")

    # Allocate memory on each GPU
    tensors = []
    for gpu_id, mem_gb in zip(gpu_ids, memory_amounts):
        tensor = occupy_gpu_memory(gpu_id, mem_gb)
        if tensor is not None:
            tensors.append(tensor)
        else:
            print(f"Warning: Failed to allocate memory on GPU {gpu_id}")

    if not tensors:
        print("Error: Failed to allocate memory on any GPU")
        sys.exit(1)

    print("\n" + "="*60)
    print("Memory allocated successfully!")
    if args.duration:
        print(f"Holding memory for {args.duration} seconds...")
    else:
        print("Holding memory indefinitely. Press Ctrl+C to release and exit.")
    print("="*60 + "\n")

    # Hold the memory
    try:
        if args.duration:
            time.sleep(args.duration)
            print("\nDuration elapsed. Releasing memory and exiting...")
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received. Releasing memory and exiting...")

    # Clean up
    del tensors
    torch.cuda.empty_cache()
    print("Memory released. Goodbye!")


if __name__ == '__main__':
    main()
