#!/bin/bash
# Manage the web UI server

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/web_ui.pid"
LOG_FILE="$PROJECT_ROOT/logs/web_ui.log"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

# Function to check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            rm "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to start server
start_server() {
    if is_running; then
        echo -e "${YELLOW}Server is already running (PID: $(cat $PID_FILE))${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Starting web UI server...${NC}"
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Start server in background
    nohup python scripts/run_web_ui.py > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    sleep 3
    
    if is_running; then
        echo -e "${GREEN}✅ Server started successfully!${NC}"
        echo -e "${GREEN}   PID: $PID${NC}"
        echo -e "${GREEN}   URL: http://localhost:8000${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to start server${NC}"
        return 1
    fi
}

# Function to stop server
stop_server() {
    if ! is_running; then
        echo -e "${YELLOW}Server is not running${NC}"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    echo -e "${YELLOW}Stopping server (PID: $PID)...${NC}"
    
    kill "$PID"
    sleep 2
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${RED}Force stopping server...${NC}"
        kill -9 "$PID"
    fi
    
    rm -f "$PID_FILE"
    echo -e "${GREEN}✓ Server stopped${NC}"
    return 0
}

# Function to restart server
restart_server() {
    echo -e "${BLUE}Restarting web UI server...${NC}"
    stop_server
    sleep 1
    start_server
}

# Function to show status
show_status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}● Server is running${NC}"
        echo -e "  PID: $PID"
        echo -e "  URL: http://localhost:8000"
        echo -e "  Logs: $LOG_FILE"
        
        # Show process info
        echo -e "\nProcess info:"
        ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,command | head -n 1
        ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,command | grep -v PID
    else
        echo -e "${RED}● Server is stopped${NC}"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}Showing last 20 lines of log:${NC}"
        tail -n 20 "$LOG_FILE"
        echo -e "\n${YELLOW}For continuous logs, run: tail -f $LOG_FILE${NC}"
    else
        echo -e "${RED}No log file found${NC}"
    fi
}

# Main script logic
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the web UI server"
        echo "  stop    - Stop the web UI server"
        echo "  restart - Restart the web UI server"
        echo "  status  - Show server status"
        echo "  logs    - Show recent logs"
        exit 1
        ;;
esac