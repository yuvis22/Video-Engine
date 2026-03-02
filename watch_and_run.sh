#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
QUEUE_DIR="$DIR/input_queue"

echo "Watching directory: $QUEUE_DIR for new .mp4 files..."
echo "Press Ctrl+C to stop."

# Simple polling loop
while true; do
    # Check if there are any mp4 files in the queue
    count=$(ls -1 "$QUEUE_DIR"/*.mp4 2>/dev/null | wc -l)
    
    if [ "$count" -gt 0 ]; then
        echo "Found $count video(s). Starting processing..."
        # Activate virtual environment and run the processor
        source "$DIR/venv/bin/activate"
        python3 "$DIR/process_video.py"
        echo "Processing complete. Resuming watch..."
    fi
    
    # Wait before checking again (e.g., 5 seconds)
    sleep 5
done
