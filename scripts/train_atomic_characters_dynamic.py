#!/usr/bin/env python3
"""
Dynamic parallel character training with GPU resource management.
Starts new characters as soon as GPU capacity becomes available.
"""

import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Set
import psutil

# Configuration
MAX_VRAM_MB = 11000  # Leave 1GB for system
VRAM_PER_CHAR_MB = 150  # Conservative estimate (observed 128 MB + buffer)
GPU_MAX_PARALLEL = MAX_VRAM_MB // VRAM_PER_CHAR_MB  # ~73 characters max (GPU limit)

# CRITICAL: System RAM constraint - each character loads full 1572 font dataset
# Brazilian Ryzen 5 with 93GB RAM (minus iGPU, KDE, system processes)
# Batch size 6 was stable, limit to 10 for safety
MAX_PARALLEL = 10  # System RAM + CPU constraint (overrides GPU calculation)

CHECK_INTERVAL = 10  # seconds

# Character sets
ALPHABET_LOWER = list('abcdefghijklmnopqrstuvwxyz')
ALPHABET_UPPER = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
DIGITS = list('0123456789')
ALL_CHARS = ALPHABET_LOWER + ALPHABET_UPPER + DIGITS

# Already successfully trained characters on FULL 1572 font set
# Batch 2 completed (verified from logs): N, O, Q, R
# All other 58 characters still need training on full font set
ALREADY_TRAINED = set('NOQR')  # Only these 4 completed on 1572 fonts

# Tracking
active_processes: Dict[str, subprocess.Popen] = {}
completed_chars: Set[str] = set()
# Only train characters that haven't been successfully trained yet
pending_chars: List[str] = [c for c in ALL_CHARS if c not in ALREADY_TRAINED]

def get_gpu_memory_used() -> int:
    """Get current GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=used_memory', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            total_used = sum(int(line.strip()) for line in result.stdout.strip().split('\n'))
            return total_used
        return 0
    except Exception as e:
        print(f"[WARNING] Could not query GPU memory: {e}")
        return 0

def start_character_training(char: str) -> subprocess.Popen:
    """Start training for a single character on FULL 1572 font set."""
    cmd = [
        '/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python',
        'scripts/train_atomic_character.py',
        '--char', char,
        '--epochs', '1500',
        # NO --fonts parameter = use all available fonts (1572 for latin)
        '--fc-only'
    ]

    print(f"[START] Character '{char}' training initiated")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
    )

    return process

def check_and_cleanup_finished():
    """Check for finished processes and clean up."""
    finished = []

    for char, process in list(active_processes.items()):
        if process.poll() is not None:  # Process finished
            returncode = process.returncode
            if returncode == 0:
                print(f"[COMPLETE] Character '{char}' finished successfully")
                completed_chars.add(char)
            else:
                print(f"[ERROR] Character '{char}' failed with code {returncode}")
                stderr = process.stderr.read().decode() if process.stderr else "No stderr"
                print(f"  Error: {stderr[:200]}")

            finished.append(char)

    # Remove finished from active tracking
    for char in finished:
        del active_processes[char]

    return len(finished)

def can_start_new_character() -> bool:
    """Check if we have capacity to start a new character."""
    # Check process count
    if len(active_processes) >= MAX_PARALLEL:
        return False

    # Check GPU memory
    used_mem = get_gpu_memory_used()
    available = MAX_VRAM_MB - used_mem

    if available < VRAM_PER_CHAR_MB:
        return False

    return True

def save_progress():
    """Save current progress to file."""
    progress = {
        'completed': list(completed_chars),
        'active': list(active_processes.keys()),
        'pending': pending_chars,
        'timestamp': time.time()
    }

    progress_file = Path('/tmp/dynamic_training_progress.json')
    progress_file.write_text(json.dumps(progress, indent=2))

def main():
    """Main training loop with dynamic resource management."""
    print(f"[INIT] Dynamic parallel training starting")
    print(f"  Total characters: {len(ALL_CHARS)}")
    print(f"  Already trained: {len(ALREADY_TRAINED)}")
    print(f"  Pending training: {len(pending_chars)}")
    print(f"  Max parallel: {MAX_PARALLEL}")
    print(f"  VRAM budget: {MAX_VRAM_MB} MB")
    print(f"  VRAM per char: {VRAM_PER_CHAR_MB} MB")

    if len(pending_chars) == 0:
        print(f"\n[COMPLETE] All {len(ALL_CHARS)} characters already trained!")
        print(f"  Trained characters: {sorted(ALREADY_TRAINED)}")
        print(f"\nNothing to do. Exiting.")
        return

    print(f"  Characters to train: {pending_chars}")
    print()

    iteration = 0

    while pending_chars or active_processes:
        iteration += 1

        # Clean up finished processes
        finished_count = check_and_cleanup_finished()

        # Start new characters if we have capacity
        started_count = 0
        while pending_chars and can_start_new_character():
            char = pending_chars.pop(0)
            process = start_character_training(char)
            active_processes[char] = process
            started_count += 1
            time.sleep(0.5)  # Small delay between starts

        # Status update
        if iteration % 6 == 0:  # Every ~60 seconds
            gpu_mem = get_gpu_memory_used()
            gpu_util = (gpu_mem / MAX_VRAM_MB) * 100

            print(f"\n[STATUS] Iteration {iteration}")
            print(f"  Active: {len(active_processes)} chars - {list(active_processes.keys())}")
            print(f"  Completed: {len(completed_chars)}/{len(ALL_CHARS)}")
            print(f"  Pending: {len(pending_chars)}")
            print(f"  GPU: {gpu_mem} MB used ({gpu_util:.1f}% of budget)")

            if started_count > 0:
                print(f"  Started {started_count} new characters this cycle")
            if finished_count > 0:
                print(f"  Finished {finished_count} characters this cycle")

        # Save progress
        save_progress()

        # Wait before next check
        time.sleep(CHECK_INTERVAL)

    print(f"\n[DONE] All characters trained!")
    print(f"  Total completed: {len(completed_chars)}")
    print(f"  Final list: {sorted(completed_chars)}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping gracefully...")
        # Kill all active processes
        for char, process in active_processes.items():
            print(f"  Terminating '{char}'...")
            process.terminate()

        save_progress()
        print(f"\n[SAVED] Progress saved. {len(completed_chars)} characters completed.")
