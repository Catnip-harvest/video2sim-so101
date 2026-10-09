#!/usr/bin/env bash
# Sample GPU VRAM used every second into $1 until killed; peak = sort -n | tail -1.
while true; do rocm-smi --showmeminfo vram 2>/dev/null | awk '/Total Used/{print $NF}' >> "$1"; sleep 1; done
