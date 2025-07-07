#!/usr/bin/env python3
"""
Script para probar el modelo Mistral 7B fine-tuneado
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json

def load_model(model_path):
    """Cargar modelo y tokenizer"""
    print(f"🤖 Cargando modelo desde: {model_path}")
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Cargar modelo base y adaptador LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generar respuesta del modelo"""
    # Formatear prompt para Mistral
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenizar
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extraer solo la respuesta generada
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response

def interactive_mode(model, tokenizer):
    """Modo interactivo para probar el modelo"""
    print("\n🎯 Modo interactivo - Escribe 'quit' para salir")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n👤 Tu pregunta: ")
            if prompt.lower() in ['quit', 'exit', 'salir']:
                break
            
            if prompt.strip():
                print("🤖 frIdA:", end=" ")
                response = generate_response(model, tokenizer, prompt)
                print(response)
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break

def batch_test(model, tokenizer, test_prompts):
    """Probar con un conjunto de prompts predefinidos"""
    print("\n🧪 Ejecutando pruebas predefinidas...")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Prueba {i}:")
        print(f"👤 Pregunta: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"🤖 frIdA: {response}")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Probar modelo Mistral 7B fine-tuneado")
    parser.add_argument("--model_path", default="./mistral_7b_frida_finetuned", 
                       help="Ruta al modelo entrenado")
    parser.add_argument("--interactive", action="store_true", 
                       help="Modo interactivo")
    parser.add_argument("--batch", action="store_true", 
                       help="Ejecutar pruebas predefinidas")
    
    args = parser.parse_args()
    
    # Cargar modelo
    try:
        model, tokenizer = load_model(args.model_path)
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Prompts de prueba predefinidos
    test_prompts = [
        "frIdA, ¿cómo optimizar el rendimiento de una aplicación web?",
        "Explícame qué es la programación orientada a objetos",
        "¿Cuáles son las mejores prácticas de seguridad en desarrollo?",
        "¿Cómo implementar autenticación JWT en una API REST?",
        "Explícame la diferencia entre SQL y NoSQL",
        "¿Qué es Docker y para qué se usa?",
        "¿Cómo funciona el machine learning?",
        "Explícame los principios SOLID de programación"
    ]
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.batch:
        batch_test(model, tokenizer, test_prompts)
    else:
        # Por defecto, ejecutar ambos
        batch_test(model, tokenizer, test_prompts)
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()
