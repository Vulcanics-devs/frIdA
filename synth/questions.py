#!/usr/bin/env python3
import json
import os
import sys
import requests
import glob
import re
from datetime import datetime
import random

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

def generate_question_with_gemini(prompt="Generate a creative trivia question"):
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

def create_question_json(question, serial_number):
    data = {
        "question": question,
        "serial_number": serial_number,
        "timestamp": datetime.now().isoformat()
    }
    return json.dumps(data, indent=2)

def get_next_serial():
    files = glob.glob("question_*.json")
    nums = [int(m.group(1)) for f in files if (m := re.search(r'question_(\d+)\.json$', f))]
    return max(nums) + 1 if nums else 1

import random

# Function to get a random seed prompt
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

def main():
    with open('prompt.txt', 'r') as f:
        master_prompt = f.read().strip()
    
    seed_prompt = get_random_seed_prompt()
    
    # Combine prompts
    if seed_prompt:
        combined_prompt = f"{master_prompt}\n\n{seed_prompt}"
    else:
        combined_prompt = master_prompt

    prompt = sys.argv[1] if len(sys.argv) > 1 else combined_prompt

    serial_number = get_next_serial()
    question = generate_question_with_gemini(prompt)

    if question:
        json_output = create_question_json(question, serial_number)
        print(json_output)
        filename = f"question_{serial_number}.json"
        with open(filename, 'w') as f:
            f.write(json_output)
        print(f"\nSaved to {filename}")
    else:
        print("Failed to generate question")

if __name__ == "__main__":
    main()
