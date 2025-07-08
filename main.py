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
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)

import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import deepspeed
from accelerate import Accelerator
import argparse
from datetime import datetime

# Optimización de memoria para H100
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

class EpochCheckpointCallback(TrainerCallback):
    """Callback personalizado para crear checkpoints detallados en cada época"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.epoch_info = []
    
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Ejecutar al final de cada época"""
        epoch = int(state.epoch)
        
        # Información de la época
        # Obtener métricas de forma segura
        train_loss = 0
        learning_rate = 0
        
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            train_loss = last_log.get("train_loss", 0)
            learning_rate = last_log.get("learning_rate", 0)
        
        epoch_data = {
            "epoch": epoch,
            "global_step": state.global_step,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_dir": f"{self.output_dir}/checkpoint-epoch-{epoch}"
        }
        
        self.epoch_info.append(epoch_data)
        
        # Guardar información de épocas
        epoch_info_path = os.path.join(self.output_dir, "epoch_info.json")
        with open(epoch_info_path, "w") as f:
            json.dump(self.epoch_info, f, indent=2)
        
        print(f"\n📊 Época {epoch} completada:")
        print(f"   📈 Loss: {epoch_data['train_loss']:.4f}")
        print(f"   📚 Learning Rate: {epoch_data['learning_rate']:.2e}")
        print(f"   💾 Checkpoint: {epoch_data['checkpoint_dir']}")
        print(f"   🕐 Timestamp: {epoch_data['timestamp']}")
        
        return control
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def parse_arguments():
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Fine-tuning de Mistral 7B con LoRA y DeepSpeed")
    parser.add_argument("--model", default="./Mistral-7B-Instruct-v0.3", 
                       help="Nombre del modelo base")
    parser.add_argument("--data", default="formatted_data.jsonl", 
                       help="Archivo de datos de entrenamiento")
    parser.add_argument("--output", default="./frida_7b", 
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
    
    # Modelo con optimizaciones para multi-GPU y DeepSpeed
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Mejor para H100
        device_map=None,  # Dejar que DeepSpeed maneje la distribución
        use_cache=False,  # Ahorra memoria durante entrenamiento
        attn_implementation="flash_attention_2",  # Flash Attention 2 para H100
        trust_remote_code=True
    )
    
    # Mover modelo a GPU principal antes de LoRA
    if torch.cuda.is_available():
        model = model.to("cuda", dtype=torch.bfloat16)
    
    # Asegurar que todos los parámetros estén en bfloat16 para FlashAttention
    for param in model.parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    
    # Preparar modelo para entrenamiento cuantizado
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def ensure_model_device_consistency(model):
    """Asegurar que todos los parámetros del modelo estén en el mismo dispositivo y tipo de datos"""
    if not torch.cuda.is_available():
        return model
    
    target_device = "cuda:0"
    target_dtype = torch.bfloat16
    
    # Verificar y mover parámetros que estén en CPU o tipo incorrecto
    for name, param in model.named_parameters():
        needs_move = param.device.type == 'cpu'
        needs_cast = param.dtype != target_dtype and param.dtype.is_floating_point
        
        if needs_move or needs_cast:
            if needs_move:
                print(f"🔄 Moviendo parámetro {name} de CPU a {target_device}")
            if needs_cast:
                print(f"🔄 Convirtiendo parámetro {name} de {param.dtype} a {target_dtype}")
            param.data = param.data.to(device=target_device, dtype=target_dtype if needs_cast else param.dtype)
    
    # Verificar buffers también
    for name, buffer in model.named_buffers():
        needs_move = buffer.device.type == 'cpu'
        needs_cast = buffer.dtype != target_dtype and buffer.dtype.is_floating_point
        
        if needs_move or needs_cast:
            if needs_move:
                print(f"🔄 Moviendo buffer {name} de CPU a {target_device}")
            if needs_cast:
                print(f"🔄 Convirtiendo buffer {name} de {buffer.dtype} a {target_dtype}")
            buffer.data = buffer.data.to(device=target_device, dtype=target_dtype if needs_cast else buffer.dtype)
    
    return model

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
    
    # Aplicar LoRA
    model = get_peft_model(model, lora_config)
    
    # Asegurar consistencia de dispositivos después de LoRA
    model = ensure_model_device_consistency(model)
    
    # Mostrar parámetros entrenables
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

def tokenize_data(file_path, tokenizer, max_length=2048):
    """Carga, tokeniza y prepara el dataset, añadiendo la columna 'labels'."""
    print(f"📂 Cargando dataset desde {file_path}...")
    dataset = load_dataset('json', data_files=file_path, split='train')
    if len(dataset) == 0:
        raise ValueError("El archivo de datos está vacío.")

    def tokenize_function(examples):
        """Función para tokenizar los textos."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )

    print("🔄 Tokenizando el dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="Tokenizando"
    )

    def add_labels(examples):
        """Añade la columna 'labels' como una copia de 'input_ids'."""
        examples["labels"] = examples["input_ids"].copy()
        return examples

    print("🏷️  Añadiendo labels al dataset...")
    tokenized_dataset = tokenized_dataset.map(
        add_labels, 
        batched=True, 
        num_proc=4, 
        desc="Añadiendo labels"
    )

    if len(tokenized_dataset) == 0:
        raise ValueError("El dataset tokenizado está vacío.")

    print(f"✅ Dataset listo: {len(tokenized_dataset)} ejemplos.")
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
    
    # 6. Usar el DataCollator estándar para Causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
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
        
        # Guardado y logging - Checkpoint en cada época
        save_strategy="epoch",  # Guardar en cada época
        save_total_limit=None,  # Mantener todos los checkpoints de época
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
        eval_strategy="no",  # Sin evaluación para maximizar velocidad
        load_best_model_at_end=False,
        
        # DeepSpeed
        deepspeed="ds_config.json",  # Configuración DeepSpeed
    )
    
    # 8. Validar data collator con una muestra del dataset
    print("🧪 Validando data collator...")
    try:
        # Tomar una pequeña muestra para probar el data collator
        test_features = [tokenized_dataset[i] for i in range(min(2, len(tokenized_dataset)))]
        test_batch = data_collator(test_features)
        
        print(f"✅ Data collator validado:")
        print(f"   - input_ids shape: {test_batch['input_ids'].shape}")
        print(f"   - attention_mask shape: {test_batch['attention_mask'].shape}")
        print(f"   - labels shape: {test_batch['labels'].shape}")
        print(f"   - input_ids dtype: {test_batch['input_ids'].dtype}")
        print(f"   - labels dtype: {test_batch['labels'].dtype}")
        
        # Verificar que labels no sean None
        if test_batch['labels'] is None:
            raise ValueError("Labels son None después del data collator")
        
        # Verificar que no haya valores NaN
        if torch.isnan(test_batch['input_ids']).any():
            raise ValueError("input_ids contienen valores NaN")
        if torch.isnan(test_batch['labels']).any():
            raise ValueError("labels contienen valores NaN")
            
        print("✅ Data collator funciona correctamente")
        
    except Exception as e:
        print(f"❌ Error validando data collator: {e}")
        raise
    
    # 9. Callback personalizado para checkpoints de época
    epoch_callback = EpochCheckpointCallback(output_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[epoch_callback],
    )
    
    # Asegurar que el modelo esté en el dispositivo correcto y tipo de datos antes del entrenamiento
    if hasattr(trainer.model, 'base_model'):
        # Para modelos PEFT, asegurar que el modelo base esté en GPU y bfloat16
        if torch.cuda.is_available():
            trainer.model.base_model.model = trainer.model.base_model.model.to("cuda", dtype=torch.bfloat16)
    
    # Verificación final: asegurar que todos los parámetros estén en bfloat16 para FlashAttention
    print("🔍 Verificando tipos de datos para FlashAttention...")
    for name, param in trainer.model.named_parameters():
        if param.dtype.is_floating_point and param.dtype != torch.bfloat16:
            print(f"⚠️ Parámetro {name} en {param.dtype}, convirtiendo a bfloat16")
            param.data = param.data.to(torch.bfloat16)
    
    for name, buffer in trainer.model.named_buffers():
        if buffer.dtype.is_floating_point and buffer.dtype != torch.bfloat16:
            print(f"⚠️ Buffer {name} en {buffer.dtype}, convirtiendo a bfloat16")
            buffer.data = buffer.data.to(torch.bfloat16)
    
    print("✅ Verificación de tipos de datos completada")
    
    # 11. Verificar datos antes del entrenamiento
    print("🔍 Verificando datos antes del entrenamiento...")
    
    # Verificar que el dataset tokenizado tenga datos
    if len(tokenized_dataset) == 0:
        raise ValueError("Dataset tokenizado está vacío")
    
    # Verificar un ejemplo del dataset
    try:
        sample = tokenized_dataset[0]
        print(f"🔍 Tipo de sample: {type(sample)}")
        print(f"🔍 Keys en sample: {list(sample.keys()) if isinstance(sample, dict) else 'No es dict'}")
        
        if not isinstance(sample, dict):
            raise ValueError(f"Dataset sample no es un diccionario, es: {type(sample)}")
        
        if "input_ids" not in sample:
            raise ValueError(f"Dataset no contiene input_ids. Keys: {list(sample.keys())}")
        
        if len(sample["input_ids"]) == 0:
            raise ValueError("input_ids están vacíos")
        
        print(f"✅ Dataset verificado: {len(tokenized_dataset)} ejemplos")
        
        # Calcular longitud promedio de forma segura
        total_length = 0
        sample_count = min(100, len(tokenized_dataset))
        
        for i in range(sample_count):
            try:
                ex = tokenized_dataset[i]
                if isinstance(ex, dict) and "input_ids" in ex:
                    total_length += len(ex["input_ids"])
                else:
                    print(f"⚠️ Ejemplo {i} no tiene formato correcto")
            except Exception as e:
                print(f"⚠️ Error accediendo ejemplo {i}: {e}")
        
        avg_length = total_length // sample_count if sample_count > 0 else 0
        print(f"📏 Longitud promedio: {avg_length} tokens")
        
    except Exception as e:
        print(f"❌ Error verificando dataset: {e}")
        print(f"🔍 Tipo de tokenized_dataset: {type(tokenized_dataset)}")
        if hasattr(tokenized_dataset, '__len__'):
            print(f"🔍 Longitud del dataset: {len(tokenized_dataset)}")
        raise
    
    # 12. Entrenar
    print("🚀 Iniciando entrenamiento distribuido...")
    print(f"📊 Batch size efectivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count()}")
    
    try:
        trainer.train()
        print("✅ Entrenamiento completado!")
        
        # Guardar modelo final
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"💾 Modelo final guardado en: {output_dir}")
        
        # Mostrar información de checkpoints
        print("\n📁 Checkpoints creados:")
        for epoch_data in epoch_callback.epoch_info:
            print(f"   Época {epoch_data['epoch']}: {epoch_data['checkpoint_dir']}")
        
        print(f"\n📊 Información detallada guardada en: {output_dir}/epoch_info.json")
        print("📊 Para cargar un checkpoint específico:")
        print(f"   from transformers import AutoModelForCausalLM")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{output_dir}/checkpoint-epoch-X')")
        
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