#!/usr/bin/env python3
"""
One-shot Galaxy buffer migration script (Kimi's implementation)
"""
import sys
import mmap
import struct
import argparse

GALAXY_EMBEDDING_SIZE_OLD = 24  # Previous size
GALAXY_EMBEDDING_SIZE = 32      # New size

def migrate_file(path):
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        old_size = GALAXY_EMBEDDING_SIZE_OLD
        new_size = GALAXY_EMBEDDING_SIZE
        n = len(mm) // old_size
        
        mm.resize(n * new_size)
        
        # Backward copy to avoid overwriting
        for i in range(n-1, -1, -1):
            old_off = i * old_size
            new_off = i * new_size
            mm[new_off:new_off+old_size] = mm[old_off:old_off+old_size]
            
            # Insert new fields
            mm[new_off+16:new_off+20] = struct.pack("I", 0)  # metadata
            mm[new_off+18:new_off+20] = struct.pack("H", 0)  # galaxy_clock
            mm[new_off+20:new_off+22] = struct.pack("H", 0)  # access_freq
            mm[new_off+24:new_off+28] = struct.pack("I", 0)  # checksum
            mm[new_off+28:new_off+32] = struct.pack("I", 0)  # reserved
        
        mm.close()
    print(f"Migrated {path} -> {n} records @ 32B")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("house_files", nargs="+", help="House files to migrate")
    args = parser.parse_args()
    
    for f in args.house_files:
        migrate_file(f)
