#!/usr/bin/env python3
"""
Script de fine-tuning para Mistral 7B Instruct optimizado para 8x H100 GPUs
Optimizado para entrenamiento distribuido con DeepSpeed y LoRA
"""

import torch
import torch.distributed as dist
import json
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import deepspeed
from accelerate import Accelerator
import argparse

# Optimización de memoria para H100
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def parse_arguments():
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Fine-tuning de Mistral 7B con LoRA y DeepSpeed")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", 
                       help="Nombre del modelo base")
    parser.add_argument("--data", default="training_data.jsonl", 
                       help="Archivo de datos de entrenamiento")
    parser.add_argument("--output", default="./mistral_7b_frida_finetuned", 
                       help="Directorio de salida")
    parser.add_argument("--epochs", type=int, default=None, 
                       help="Número de épocas (sobrescribe cálculo automático)")
    parser.add_argument("--batch-size", type=int, default=8, 
                       help="Batch size por dispositivo")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                       help="Learning rate")
    return parser.parse_args()

def check_gpu():
    """Verificar disponibilidad de GPUs"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPUs disponibles: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        return True
    else:
        print("❌ No se detectó GPU CUDA")
        return False

def load_jsonl_data(file_path, max_samples=None):
    """Cargar datos del archivo JSONL con formato Mistral"""
    data = []
    print(f"📁 Cargando datos de: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                if 'text' in item:
                    # El formato ya viene correcto desde el JSONL
                    # Formato: <s>[INST] ... [/INST] ... </s>
                    data.append({"text": item['text']})
                else:
                    print(f"⚠️  Línea {i+1} no tiene campo 'text'")
            except json.JSONDecodeError:
                print(f"❌ Error en línea {i+1}")
    
    print(f"✅ Cargados {len(data)} ejemplos")
    return data

def setup_model_and_tokenizer(args):
    """Configurar modelo Mistral 7B Instruct y tokenizer"""
    model_name = args.model
    print(f"🤖 Cargando modelo: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=True
    )
    
    # Agregar pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Modelo con optimizaciones para multi-GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Mejor para H100
        device_map="auto",  # Distribución automática entre GPUs
        use_cache=False,  # Ahorra memoria durante entrenamiento
        attn_implementation="flash_attention_2",  # Flash Attention 2 para H100
        trust_remote_code=True
    )
    
    # Preparar modelo para entrenamiento cuantizado
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model):
    """Configurar LoRA para entrenamiento eficiente en Mistral 7B"""
    lora_config = LoraConfig(
        r=64,  # Rank más alto para mejor calidad con H100
        lora_alpha=128,  # Alpha escalado
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,  # Dropout más bajo para mejor convergencia
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["lm_head", "embed_tokens"]  # Entrenar también embeddings
    )
    
    model = get_peft_model(model, lora_config)
    print("📊 Parámetros del modelo:")
    model.print_trainable_parameters()
    
    return model

def calculate_optimal_epochs(dataset_size):
    """Calcular número óptimo de épocas según el tamaño del dataset"""
    if dataset_size < 1000:
        epochs = 5  # Dataset muy pequeño, necesita más pasadas
        print(f"📊 Dataset pequeño ({dataset_size} ejemplos) → {epochs} épocas")
    elif dataset_size < 10000:
        epochs = 4  # Dataset pequeño-mediano
        print(f"📊 Dataset pequeño-mediano ({dataset_size} ejemplos) → {epochs} épocas")
    elif dataset_size < 50000:
        epochs = 3  # Dataset mediano (óptimo)
        print(f"📊 Dataset mediano ({dataset_size} ejemplos) → {epochs} épocas")
    elif dataset_size < 100000:
        epochs = 2  # Dataset grande
        print(f"📊 Dataset grande ({dataset_size} ejemplos) → {epochs} épocas")
    else:
        epochs = 1  # Dataset muy grande
        print(f"📊 Dataset muy grande ({dataset_size} ejemplos) → {epochs} época")
    
    print(f"💡 Razón: Evitar overfitting y optimizar tiempo de entrenamiento")
    return epochs

def tokenize_data(data, tokenizer, max_length=2048):
    """Tokenizar los datos con longitud optimizada para Mistral"""
    def tokenize_function(examples):
        # Tokenizar con labels para entrenamiento causal
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Padding dinámico en data collator
            max_length=max_length,
            return_overflowing_tokens=False,
            add_special_tokens=False  # Ya están en el texto
        )
        
        # Para entrenamiento causal, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=8,  # Paralelización para velocidad
        desc="Tokenizando datos"
    )
    
    return tokenized_dataset

def main(args):
    # 1. Verificar GPUs
    if not check_gpu():
        return
    
    # 2. Configuración desde argumentos
    jsonl_path = args.data
    output_dir = args.output
    max_samples = None  # Usar todos los datos disponibles
    
    # 3. Cargar datos
    if not os.path.exists(jsonl_path):
        print(f"❌ No se encontró el archivo: {jsonl_path}")
        print("📝 Crea un archivo JSONL con formato:")
        print('{"text": "<s>[INST] pregunta [/INST] respuesta </s>"}')
        return
    
    data = load_jsonl_data(jsonl_path, max_samples=max_samples)
    if not data:
        print("❌ No se cargaron datos válidos")
        return
    
    # 4. Configurar modelo
    model, tokenizer = setup_model_and_tokenizer(args)
    model = setup_lora(model)
    
    # 5. Tokenizar datos
    print("🔄 Tokenizando datos...")
    tokenized_dataset = tokenize_data(data, tokenizer)
    
    # Calcular épocas óptimas según el tamaño del dataset
    dataset_size = len(data)
    if args.epochs:
        optimal_epochs = args.epochs
        print(f"🔄 Usando épocas manuales: {optimal_epochs}")
    else:
        optimal_epochs = calculate_optimal_epochs(dataset_size)
    
    # 6. Data collator optimizado
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Optimización para H100
    )
    
    # 7. Configuración de entrenamiento optimizada para 8x H100
    training_args = TrainingArguments(
        output_dir=output_dir,
        # Épocas calculadas automáticamente según el tamaño del dataset
        # Puedes sobrescribir manualmente si es necesario
        num_train_epochs=optimal_epochs,
        per_device_train_batch_size=args.batch_size,  # Configurable
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,  # Batch efectivo = batch_size*8*2
        learning_rate=args.learning_rate,  # Configurable
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        
        # Guardado y logging
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        
        # Optimizaciones de memoria y velocidad
        bf16=True,  # BFloat16 para H100
        tf32=True,  # TensorFloat-32 para H100
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        
        # Entrenamiento distribuido
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        
        # Optimizaciones adicionales
        gradient_checkpointing=True,  # Ahorra memoria
        max_grad_norm=1.0,
        
        # Reporting
        report_to=["tensorboard"],
        run_name="frIdA-7b",
        
        # Evaluación
        evaluation_strategy="no",  # Sin evaluación para maximizar velocidad
        load_best_model_at_end=False,
        
        # DeepSpeed
        deepspeed="ds_config.json",  # Configuración DeepSpeed
    )
    
    # 8. Trainer con optimizaciones
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 9. Entrenar
    print("🚀 Iniciando entrenamiento distribuido...")
    print(f"📊 Batch size efectivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count()}")
    
    try:
        trainer.train()
        print("✅ Entrenamiento completado!")
        
        # Guardar modelo final
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"💾 Modelo guardado en: {output_dir}")
        
    except torch.cuda.OutOfMemoryError:
        print("❌ Error de memoria GPU!")
        print("💡 Intenta reducir per_device_train_batch_size")
    except Exception as e:
        print(f"❌ Error durante entrenamiento: {e}")

def test_model(model_path="./frIdA-7b"):
    """Función para probar el modelo Mistral entrenado"""
    print("🧪 Probando modelo entrenado...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Ejemplos de inferencia con formato Mistral
    test_prompts = [
        "<s>[INST] frIdA, ¿cómo optimizar el rendimiento de una aplicación web? [/INST]",
        "<s>[INST] Explícame qué es la programación orientada a objetos [/INST]",
        "<s>[INST] ¿Cuáles son las mejores prácticas de seguridad en desarrollo? [/INST]"
    ]
    
    for i, test_prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Prueba {i}:")
        print(f"Prompt: {test_prompt}")
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
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
        
        print(f"🤖 Respuesta: {response}")
        print("-" * 80)

if __name__ == "__main__":
    # Parsear argumentos
    args = parse_arguments()
    
    print("🔥 Fine-tuning Mistral 7B Instruct - frIdA")
    print("🚀 Optimizado para 8x H100 GPUs")
    print(f"🤖 Modelo: {args.model}")
    print(f"📁 Datos: {args.data}")
    print(f"💾 Salida: {args.output}")
    if args.epochs:
        print(f"🔄 Épocas (manual): {args.epochs}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🎯 Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Ejecutar entrenamiento
    main(args)
    
    # Opcional: probar el modelo después del entrenamiento
    # test_model(args.output)