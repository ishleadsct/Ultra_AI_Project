#!/data/data/com.termux/files/usr/bin/bash

# Kill any old session first
tmux kill-session -t ultra 2>/dev/null || true

# Ensure __init__.py exists
touch ~/Ultra_AI/utils/__init__.py
touch ~/Ultra_AI/core/__init__.py
touch ~/Ultra_AI/ui/__init__.py

# Start Ultra AI tmux session with multiple windows
tmux new-session -d -s ultra

# Window 0: UI
tmux rename-window -t ultra:0 'UI'
tmux send-keys -t ultra:0 "export PYTHONPATH=$HOME/Ultra_AI && python $HOME/Ultra_AI/ui/server.py" C-m

# Window 1: Orchestrator
tmux new-window -t ultra:1 -n 'Orchestrator'
tmux send-keys -t ultra:1 "export PYTHONPATH=$HOME/Ultra_AI && python $HOME/Ultra_AI/core/orchestrator.py" C-m

# Window 2: Librarian
tmux new-window -t ultra:2 -n 'Librarian'
tmux send-keys -t ultra:2 "export PYTHONPATH=$HOME/Ultra_AI && python $HOME/Ultra_AI/core/librarian.py" C-m

echo "Ultra AI started. Attach to tmux with: tmux attach -t ultra"
