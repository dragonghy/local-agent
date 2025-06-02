#!/bin/bash
# Start/restart the web UI server in background

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/web_ui.pid"
LOG_FILE="$PROJECT_ROOT/logs/web_ui.log"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

echo -e "${YELLOW}Local LLM Web UI Manager${NC}"
echo "========================"

# Function to check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            # PID file exists but process is dead
            rm "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to stop server
stop_server() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${YELLOW}Stopping existing server (PID: $PID)...${NC}"
        kill "$PID"
        sleep 2
        
        # Force kill if still running
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${RED}Force stopping server...${NC}"
            kill -9 "$PID"
        fi
        
        rm -f "$PID_FILE"
        echo -e "${GREEN}✓ Server stopped${NC}"
    fi
}

# Stop existing server if running
if is_running; then
    echo -e "${YELLOW}Server is already running${NC}"
    stop_server
fi

# Start new server
echo -e "${YELLOW}Starting web UI server...${NC}"

# Activate virtual environment and start server
cd "$PROJECT_ROOT"
source venv/bin/activate

# Start server in background and redirect output to log file
nohup python scripts/run_web_ui.py > "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo $PID > "$PID_FILE"

# Wait a moment to check if server started successfully
sleep 3

if is_running; then
    echo -e "${GREEN}✅ Server started successfully!${NC}"
    echo -e "${GREEN}   PID: $PID${NC}"
    echo -e "${GREEN}   URL: http://localhost:8000${NC}"
    echo -e "${GREEN}   Logs: tail -f $LOG_FILE${NC}"
    echo ""
    echo "Commands:"
    echo "  Stop server:   $0 stop"
    echo "  Server status: $0 status"
    echo "  View logs:     tail -f $LOG_FILE"
else
    echo -e "${RED}❌ Failed to start server${NC}"
    echo -e "${RED}Check logs: $LOG_FILE${NC}"
    exit 1
fi