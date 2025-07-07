#!/bin/bash

# runit.sh - Run multiple instances of display_generator.sh concurrently

SCRIPT_NAME="display_generator.sh"
PIDS_DIR="pids"
LOGS_DIR="logs"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

start_generator() {
    local num_instances=${1:-1}
    
    echo -e "${BLUE}Starting $num_instances instances of $SCRIPT_NAME...${NC}"
    
    # Create directories
    mkdir -p "$PIDS_DIR" "$LOGS_DIR"
    
    # Stop any existing processes first
    stop_generator
    
    # Start multiple instances
    for ((i=1; i<=num_instances; i++)); do
        local log_file="$LOGS_DIR/output_$i.log"
        local pid_file="$PIDS_DIR/generator_$i.pid"
        
        echo -e "${BLUE}Starting instance $i...${NC}"
        
        # Start the instance (all instances use the same output file)
        nohup ./"$SCRIPT_NAME" > "$log_file" 2>&1 &
        local pid=$!
        echo "$pid" > "$pid_file"
        
        echo -e "${GREEN}Instance $i started with PID: $pid${NC}"
        sleep 0.5
    done
    
    echo -e "${GREEN}All $num_instances instances started${NC}"
    echo -e "${BLUE}Logs in: $LOGS_DIR/${NC}"
    echo -e "${BLUE}PIDs in: $PIDS_DIR/${NC}"
}

status_generator() {
    local running=0
    local total=0
    
    if [[ -d "$PIDS_DIR" ]]; then
        for pid_file in "$PIDS_DIR"/generator_*.pid; do
            if [[ -f "$pid_file" ]]; then
                total=$((total + 1))
                local pid=$(cat "$pid_file")
                if kill -0 "$pid" 2>/dev/null; then
                    running=$((running + 1))
                    echo -e "${GREEN}Instance $(basename "$pid_file" .pid | cut -d_ -f2) running (PID: $pid)${NC}"
                else
                    echo -e "${RED}Instance $(basename "$pid_file" .pid | cut -d_ -f2) not running (stale PID file)${NC}"
                    rm -f "$pid_file"
                fi
            fi
        done
    fi
    
    if [[ $total -eq 0 ]]; then
        echo -e "${RED}No instances found${NC}"
        return 1
    else
        echo -e "${BLUE}Status: $running/$total instances running${NC}"
        return $([[ $running -eq $total ]] && echo 0 || echo 1)
    fi
}

stop_generator() {
    local stopped=0
    
    # First kill all processes from PID files
    if [[ -d "$PIDS_DIR" ]]; then
        for pid_file in "$PIDS_DIR"/generator_*.pid; do
            if [[ -f "$pid_file" ]]; then
                local pid=$(cat "$pid_file")
                local instance=$(basename "$pid_file" .pid | cut -d_ -f2)
                
                if kill -0 "$pid" 2>/dev/null; then
                    echo -e "${YELLOW}Stopping instance $instance (PID: $pid)${NC}"
                    kill "$pid"
                    sleep 1
                    if kill -0 "$pid" 2>/dev/null; then
                        echo -e "${RED}Force killing instance $instance${NC}"
                        kill -9 "$pid"
                    fi
                    stopped=$((stopped + 1))
                fi
                rm -f "$pid_file"
            fi
        done
    fi
    
    # Kill any remaining processes by name
    echo -e "${YELLOW}Killing any remaining generator processes...${NC}"
    pkill -f "display_generator" 2>/dev/null && stopped=$((stopped + 1))
    pkill -f "data_generator.py" 2>/dev/null && stopped=$((stopped + 1))
    
    # Clean up generated script files (no longer needed since all use same script)
    # rm -f "${SCRIPT_NAME}_"*
    
    # Clean up directories if empty
    [[ -d "$PIDS_DIR" ]] && rmdir "$PIDS_DIR" 2>/dev/null || true
    
    if [[ $stopped -eq 0 ]]; then
        echo -e "${YELLOW}No running instances found${NC}"
    else
        echo -e "${GREEN}Stopped all generator processes${NC}"
    fi
}

tail_log() {
    local instance=${1:-1}
    local log_file="$LOGS_DIR/output_$instance.log"
    
    if [[ -f "$log_file" ]]; then
        echo -e "${BLUE}Tailing log file for instance $instance (Ctrl+C to exit)...${NC}"
        tail -f "$log_file"
    else
        echo -e "${RED}Log file for instance $instance not found${NC}"
        echo -e "${BLUE}Available logs:${NC}"
        ls -1 "$LOGS_DIR"/ 2>/dev/null || echo -e "${RED}No logs directory found${NC}"
    fi
}

case "$1" in
    start)
        start_generator "$2"
        ;;
    stop)
        stop_generator
        ;;
    status)
        status_generator
        ;;
    restart)
        stop_generator
        sleep 1
        start_generator "$2"
        ;;
    log)
        tail_log "$2"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart|log} [number|instance]"
        echo "  start [N]   - Start N instances of the generator (default: 1)"
        echo "  stop        - Stop all generator instances"
        echo "  status      - Check status of all instances"
        echo "  restart [N] - Stop and start N instances"
        echo "  log [N]     - Tail the output log for instance N (default: 1)"
        echo ""
        echo "Examples:"
        echo "  $0 start 8    - Start 8 concurrent instances"
        echo "  $0 start 64   - Start 64 concurrent instances"
        echo "  $0 log 3      - View log for instance 3"
        exit 1
        ;;
esac
