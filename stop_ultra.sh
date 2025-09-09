#!/data/data/com.termux/files/usr/bin/bash

# Kill the Ultra AI tmux session
tmux kill-session -t ultra 2>/dev/null || true
echo "Ultra AI stopped."
