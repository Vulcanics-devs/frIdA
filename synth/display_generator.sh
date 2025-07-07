#!/bin/bash

set -e

# Configuration
TARGET_LINES=250000
OUTPUT_FILE="synthetic_data.jsonl"
BATCH_SIZE=100
PYTHON_SCRIPT="data_generator.py"
LOG_FILE="generation.log"

# Colors for fancy output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# Unicode symbols
CHECKMARK="✓"
CROSS="✗"
ARROW="→"
STAR="★"
GEAR="⚙"

# Global variables for tracking
declare -A category_counts
total_generated=0
start_time=$(date +%s)

print_header() {
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}${STAR} ${CYAN}Synthetic Data Generation Script${WHITE} ${STAR}${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Target Lines:${NC} ${WHITE}${TARGET_LINES}${NC}"
    echo -e "${BLUE}Output File: ${NC} ${WHITE}${OUTPUT_FILE}${NC}"
    echo -e "${BLUE}Batch Size:  ${NC} ${WHITE}${BATCH_SIZE}${NC}"
    echo -e "${BLUE}Categories:  ${NC} ${WHITE}10 types (culture, sports, music, politics, business, tech, code, emotional, social, casual)${NC}"
    echo -e "${YELLOW}📊 Live ASCII bar chart will show category distribution during generation${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
}

get_line_count() {
    if [[ -f "$OUTPUT_FILE" ]]; then
        wc -l < "$OUTPUT_FILE"
    else
        echo "0"
    fi
}

print_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "\r${CYAN}Progress: ${NC}["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] ${WHITE}%d%%${NC} (${GREEN}%d${NC}/${WHITE}%d${NC} lines)" "$percentage" "$current" "$total"
}

estimate_time() {
    local current=$1
    local total=$2
    local elapsed=$3
    
    if [[ $current -gt 0 ]]; then
        local rate=$((current / elapsed))
        local remaining=$((total - current))
        local eta=$((remaining / rate))
        
        if [[ $eta -gt 3600 ]]; then
            printf "${YELLOW}ETA: %dh %dm${NC}" $((eta / 3600)) $(((eta % 3600) / 60))
        elif [[ $eta -gt 60 ]]; then
            printf "${YELLOW}ETA: %dm %ds${NC}" $((eta / 60)) $((eta % 60))
        else
            printf "${YELLOW}ETA: %ds${NC}" $eta
        fi
    fi
}

display_live_data() {
    local question="$1"
    local response="$2"
    local category="$3"
    local timestamp="$4"
    
    echo -e "\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}${STAR} ${CYAN}Latest Generated Pair${WHITE} ${STAR}${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    
    echo -e "${BLUE}❓ QUESTION:${NC}"
    echo -e "${YELLOW}${question}${NC}" | fold -w 80 -s
    echo
    
    echo -e "${BLUE}🤖 frIdA RESPONSE:${NC}"
    echo -e "${GREEN}${response}${NC}" | fold -w 80 -s
    echo
    
    echo -e "${BLUE}📂 Category:${NC} ${WHITE}${category}${NC}"
    echo -e "${BLUE}⏰ Timestamp:${NC} ${WHITE}${timestamp}${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
}

create_ascii_bars() {
    local total=$1
    
    if [[ $total -eq 0 ]]; then
        return
    fi
    
    echo -e "\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}${STAR} ${CYAN}Category Distribution${WHITE} ${STAR}${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    
    local categories=("culture" "sports" "music" "politics" "business" "technology" "code" "emotional" "social" "casual")
    local colors=("${RED}" "${GREEN}" "${YELLOW}" "${BLUE}" "${PURPLE}" "${CYAN}" "${WHITE}" "${RED}" "${GREEN}" "${YELLOW}")
    
    for i in "${!categories[@]}"; do
        local category="${categories[$i]}"
        local count="${category_counts[$category]:-0}"
        local color="${colors[$i]}"
        
        if [[ $count -gt 0 ]]; then
            local percentage=$((count * 100 / total))
            local bar_length=$((percentage / 3))
            [[ $bar_length -gt 30 ]] && bar_length=30
            
            printf "%-12s %s%3d%% " "$category" "$color" "$percentage"
            printf "%*s" "$bar_length" | tr ' ' '█'
            printf " %s(%d)${NC}\n" "$color" "$count"
        fi
    done
    
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
}

update_category_count() {
    local category="$1"
    category_counts["$category"]=$((${category_counts["$category"]:-0} + 1))
    total_generated=$((total_generated + 1))
}

show_generation_stats() {
    local elapsed=$(($(date +%s) - start_time))
    local rate=$((total_generated / elapsed))
    [[ $elapsed -eq 0 ]] && rate=0
    
    echo -e "\n${BLUE}📈 Generation Statistics:${NC}"
    echo -e "${BLUE}Total Generated: ${GREEN}${total_generated}${NC}"
    echo -e "${BLUE}⚡ Rate: ${YELLOW}${rate}/min${NC}"
    echo -e "${BLUE}⏱️ Elapsed: ${CYAN}${elapsed}s${NC}"
}

process_json_output() {
    local json_line="$1"
    
    # Parse JSON using python (more reliable than jq)
    local question=$(echo "$json_line" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data.get('question', ''))
except: pass
")
    
    local response=$(echo "$json_line" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data.get('response', ''))
except: pass
")
    
    local category=$(echo "$json_line" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data.get('category', 'casual'))
except: pass
")
    
    local timestamp=$(echo "$json_line" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data.get('timestamp', ''))
except: pass
")
    
    if [[ -n "$question" && -n "$response" ]]; then
        update_category_count "$category"
        display_live_data "$question" "$response" "$category" "$timestamp"
        create_ascii_bars "$total_generated"
        show_generation_stats
        
        # Save to output file
        echo "$json_line" >> "$OUTPUT_FILE"
    fi
}

check_prerequisites() {
    echo -e "${GEAR} ${BLUE}Checking prerequisites...${NC}"
    
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        echo -e "${CROSS} ${RED}Error: $PYTHON_SCRIPT not found${NC}"
        exit 1
    fi
    
    if [[ ! -f "prompt.txt" ]]; then
        echo -e "${CROSS} ${RED}Error: prompt.txt not found${NC}"
        exit 1
    fi
    
    if [[ ! -f "seed.txt" ]]; then
        echo -e "${CROSS} ${RED}Error: seed.txt not found${NC}"
        exit 1
    fi
    
    if [[ ! -f ".env" ]]; then
        echo -e "${CROSS} ${RED}Error: .env file not found${NC}"
        exit 1
    fi
    
    echo -e "${CHECKMARK} ${GREEN}Prerequisites check passed${NC}"
}

cleanup() {
    echo -e "\n${YELLOW}${ARROW} Interrupted by user${NC}"
    local current_lines=$(get_line_count)
    echo -e "${BLUE}Current progress: ${GREEN}${current_lines}${NC} lines generated"
    echo -e "${CYAN}Output saved to: ${WHITE}${OUTPUT_FILE}${NC}"
    exit 0
}

main() {
    print_header
    check_prerequisites
    
    trap cleanup SIGINT SIGTERM
    
    echo "$(date): Starting synthetic data generation" > "$LOG_FILE"
    
    local current_lines=$(get_line_count)
    local batch_count=0
    local consecutive_failures=0
    local max_failures=5
    
    echo -e "${GEAR} ${BLUE}Starting generation...${NC}"
    echo -e "${BLUE}Starting with ${GREEN}${current_lines}${NC} existing lines${NC}"
    
    while [[ $current_lines -lt $TARGET_LINES ]]; do
        local remaining=$((TARGET_LINES - current_lines))
        local current_batch_size=$((remaining < BATCH_SIZE ? remaining : BATCH_SIZE))
        
        echo -e "\n${ARROW} ${CYAN}Generating batch $((batch_count + 1)) (${current_batch_size} pairs)...${NC}"
        
        # Generate data and process each JSON line
        python3 "$PYTHON_SCRIPT" "$current_batch_size" 2>>"$LOG_FILE" | while IFS= read -r json_line; do
            if [[ -n "$json_line" ]]; then
                process_json_output "$json_line"
            fi
        done
        
        local new_lines=$(get_line_count)
        local generated=$((new_lines - current_lines))
        
        if [[ $generated -gt 0 ]]; then
            echo -e "${CHECKMARK} ${GREEN}Generated ${generated} lines${NC}"
            current_lines=$new_lines
            batch_count=$((batch_count + 1))
            consecutive_failures=0
            
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_time))
            print_progress "$current_lines" "$TARGET_LINES"
            echo -n " | "
            estimate_time "$current_lines" "$TARGET_LINES" "$elapsed"
            echo
        else
            echo -e "${CROSS} ${RED}No lines generated in this batch${NC}"
            consecutive_failures=$((consecutive_failures + 1))
        fi
        
        if [[ $consecutive_failures -ge $max_failures ]]; then
            echo -e "${CROSS} ${RED}Too many consecutive failures. Stopping.${NC}"
            break
        fi
        
        sleep 1
    done
    
    local end_time=$(date +%s)
    local total_elapsed=$((end_time - start_time))
    
    echo -e "\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}${STAR} ${GREEN}Generation Complete${WHITE} ${STAR}${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Total Lines Generated: ${GREEN}$(get_line_count)${NC}"
    echo -e "${BLUE}Total Time Elapsed:    ${CYAN}${total_elapsed}s${NC}"
    echo -e "${BLUE}Total Batches:         ${WHITE}${batch_count}${NC}"
    echo -e "${BLUE}Output File:           ${WHITE}${OUTPUT_FILE}${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    
    create_ascii_bars "$total_generated"
    
    echo "$(date): Generation completed. Total lines: $(get_line_count)" >> "$LOG_FILE"
}

main "$@"