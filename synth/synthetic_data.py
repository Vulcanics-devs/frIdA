#!/usr/bin/env python3
import json
import os
import sys
import requests
import glob
import re
from datetime import datetime
import random
from collections import defaultdict
import threading
import time
import math

# Load .env file manually
def load_env_file(path=".env"):
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            os.environ[key.strip()] = value.strip()

# Initialize environment
load_env_file()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"

def generate_with_gemini(prompt):
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in environment.")
        return None

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    { "text": prompt }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            params={"key": GEMINI_API_KEY},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text'].strip()
    except requests.RequestException as e:
        print(f"HTTP Error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Unexpected API response: {e}")
        print(f"Raw response: {response.text}")
        return None

def get_random_seed_prompt():
    try:
        with open('seed.txt', 'r') as f:
            content = f.read().strip()
            if not content:
                return ""
            seeds = [seed.strip() for seed in content.split(',') if seed.strip()]
            return random.choice(seeds) if seeds else ""
    except FileNotFoundError:
        return ""

# Global category tracker
category_counts = defaultdict(int)
category_lock = threading.Lock()

# Available categories
CATEGORIES = [
    "culture", "sports", "music", "politics", 
    "business", "technology", "code", "emotional", 
    "social", "casual"
]

def classify_question(question):
    """Classify a question into one of the predefined categories"""
    classification_prompt = f"""Classify this question into exactly ONE of these categories:
- culture: Questions about traditions, history, art, literature, customs
- sports: Questions about any sport, exercise, fitness, athletes
- music: Questions about music, instruments, songs, artists, composition
- politics: Questions about government, laws, elections, political figures
- business: Questions about economics, finance, startups, investments, entrepreneurship
- technology: Questions about tech, gadgets, software, AI, innovation
- code: Questions about programming, coding, software development, algorithms
- emotional: Questions about feelings, relationships, mental health, personal advice
- social: Questions about social interactions, networking, community
- casual: Everyday questions, simple requests, general conversation

Question: "{question}"

Respond with ONLY the category name, nothing else."""

    category = generate_with_gemini(classification_prompt)
    if category and category.lower().strip() in CATEGORIES:
        return category.lower().strip()
    return "casual"  # Default fallback

def create_ascii_pie_chart(category_counts, total_width=50):
    """Create an ASCII pie chart representation"""
    if not category_counts:
        return ""
    
    total = sum(category_counts.values())
    if total == 0:
        return ""
    
    # Colors for different categories (terminal colors)
    colors = {
        'culture': '\033[91m',    # Red
        'sports': '\033[92m',     # Green
        'music': '\033[93m',      # Yellow
        'politics': '\033[94m',   # Blue
        'business': '\033[95m',   # Magenta
        'technology': '\033[96m', # Cyan
        'code': '\033[97m',       # White
        'emotional': '\033[90m',  # Dark Gray
        'social': '\033[35m',     # Purple
        'casual': '\033[33m'      # Orange
    }
    reset = '\033[0m'
    
    # Create the pie chart
    chart = []
    chart.append("┌" + "─" * (total_width + 2) + "┐")
    chart.append("│" + " " * (total_width + 2) + "│")
    
    # Create the visual representation
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        bar_length = int((count / total) * total_width)
        color = colors.get(category, '\033[37m')  # Default to white
        
        # Create the bar
        bar = color + "█" * bar_length + reset
        padding = " " * (total_width - bar_length)
        
        chart.append(f"│ {bar}{padding} │ {category}: {count} ({percentage:.1f}%)")
    
    chart.append("│" + " " * (total_width + 2) + "│")
    chart.append("└" + "─" * (total_width + 2) + "┘")
    
    return "\n".join(chart)

def update_ascii_chart():
    """Update the ASCII chart in terminal"""
    last_total = 0
    
    while True:
        with category_lock:
            current_total = sum(category_counts.values())
            
            if current_total > 0 and current_total != last_total:
                # Clear screen and show updated chart
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🎯 Live Category Distribution")
                print("=" * 60)
                print(create_ascii_pie_chart(category_counts))
                print(f"\nTotal Questions Generated: {current_total}")
                print("=" * 60)
                print("Press Ctrl+C to stop...")
                
                last_total = current_total
        
        time.sleep(3)  # Update every 3 seconds

def generate_synthetic_data_pair():
    # Load the master prompt for generating questions
    with open('prompt.txt', 'r') as f:
        master_prompt = f.read().strip()
    
    seed_prompt = get_random_seed_prompt()
    
    # Combine prompts for question generation
    if seed_prompt:
        question_prompt = f"{master_prompt}\n\n{seed_prompt}"
    else:
        question_prompt = master_prompt

    # Generate the question
    question = generate_with_gemini(question_prompt)
    if not question:
        return None
    
    # Classify the question
    category = classify_question(question)
    
    # Update category counts
    with category_lock:
        category_counts[category] += 1
    
    # Create the instruction format for the question
    instruction_prompt = f"<s>[INST] {question} [/INST]"
    
    # Generate the response by sending the question to the model
    response = generate_with_gemini(question)
    if not response:
        return None
    
    # Format as JSONL with category
    synthetic_pair = {
        "prompt": instruction_prompt,
        "response": response,
        "category": category
    }
    
    return synthetic_pair

def main():
    num_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    output_file = sys.argv[2] if len(sys.argv) > 2 else "synthetic_data.jsonl"
    show_chart = len(sys.argv) > 3 and sys.argv[3].lower() == "--chart"
    
    print(f"Generating {num_pairs} synthetic data pairs...")
    
    # Start the ASCII chart updater in a separate thread if requested
    chart_thread = None
    if show_chart:
        print("🎯 Starting live ASCII pie chart visualization...")
        chart_thread = threading.Thread(target=update_ascii_chart, daemon=True)
        chart_thread.start()
    
    # Load existing data to continue counting
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'category' in data:
                        with category_lock:
                            category_counts[data['category']] += 1
                except json.JSONDecodeError:
                    continue
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in range(num_pairs):
            print(f"Generating pair {i+1}/{num_pairs}...")
            
            synthetic_pair = generate_synthetic_data_pair()
            if synthetic_pair:
                f.write(json.dumps(synthetic_pair, ensure_ascii=False) + '\n')
                print(f"✓ Generated pair {i+1} - Category: {synthetic_pair['category']}")
                
                # Print current distribution every 10 pairs
                if (i + 1) % 10 == 0:
                    print("\n📊 Current category distribution:")
                    total = sum(category_counts.values())
                    for cat, count in sorted(category_counts.items()):
                        percentage = (count / total) * 100
                        print(f"  {cat}: {count} ({percentage:.1f}%)")
                    print()
            else:
                print(f"✗ Failed to generate pair {i+1}")
    
    print(f"\nSynthetic data saved to {output_file}")
    
    # Final category distribution
    print("\n🎯 Final category distribution:")
    total = sum(category_counts.values())
    for cat, count in sorted(category_counts.items()):
        percentage = (count / total) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    if show_chart:
        print("\n📈 ASCII chart running in background. Check terminal for live updates!")
        time.sleep(2)  # Let chart display briefly

if __name__ == "__main__":
    main()