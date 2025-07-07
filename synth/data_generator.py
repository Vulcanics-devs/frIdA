#!/usr/bin/env python3
import json
import os
import sys
import requests
import time
import random
from datetime import datetime

def load_env_file(path=".env"):
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            os.environ[key.strip()] = value.strip()

load_env_file()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"

CATEGORIES = [
    "culture", "sports", "music", "politics", 
    "business", "technology", "code", "emotional", 
    "social", "casual"
]

def generate_with_gemini(prompt):
    if not GEMINI_API_KEY:
        return None

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            params={"key": GEMINI_API_KEY},
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text'].strip()
    except:
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

def get_frida_prompt():
    try:
        with open('frida.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Responde de manera útil y profesional."

def classify_question(question):
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
    return "casual"

def generate_synthetic_data_pair():
    try:
        with open('prompt.txt', 'r') as f:
            master_prompt = f.read().strip()
    except FileNotFoundError:
        return None
    
    seed_prompt = get_random_seed_prompt()
    
    if seed_prompt:
        question_prompt = f"{master_prompt}\n\n{seed_prompt}"
    else:
        question_prompt = master_prompt

    question = generate_with_gemini(question_prompt)
    if not question:
        return None
    
    category = classify_question(question)
    
    frida_prompt = get_frida_prompt()
    response_prompt = f"{frida_prompt}\n\nPregunta: {question}\n\nResponde como frIdA:"
    response = generate_with_gemini(response_prompt)
    if not response:
        return None
    
    instruction_prompt = f"<s>[INST] {question} [/INST]"
    
    synthetic_pair = {
        "prompt": instruction_prompt,
        "response": response,
        "category": category,
        "question": question,
        "timestamp": datetime.now().isoformat()
    }
    
    return synthetic_pair

def main():
    num_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    for i in range(num_pairs):
        synthetic_pair = generate_synthetic_data_pair()
        if synthetic_pair:
            print(json.dumps(synthetic_pair, ensure_ascii=False))
            sys.stdout.flush()
        else:
            print(json.dumps({"error": "Failed to generate pair"}), file=sys.stderr)

if __name__ == "__main__":
    main()