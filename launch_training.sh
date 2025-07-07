#!/bin/bash

# Script para lanzar entrenamiento distribuido de Mistral 7B con 8 GPUs H100
# Uso: bash launch_training.sh

echo "🚀 Iniciando entrenamiento distribuido Mistral 7B - frIdA"
echo "🔧 Configuración: 8x H100 GPUs con DeepSpeed ZeRO Stage 2"
echo "=" * 60

# Paso 1: Crear y activar entorno virtual
echo "🐍 Configurando entorno virtual..."
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Error: No se pudo crear el entorno virtual"
        echo "💡 Asegúrate de tener python-venv instalado: sudo apt install python3-venv"
        exit 1
    fi
fi

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Error: No se pudo activar el entorno virtual"
    exit 1
fi

echo "✅ Entorno virtual activado: $(which python)"

# Paso 2: Verificar e instalar dependencias
echo "📦 Verificando dependencias..."
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python no está disponible en el entorno virtual"
    exit 1
fi

# Actualizar pip
echo "🔄 Actualizando pip..."
pip install --upgrade pip

# Verificar si torch está instalado
if ! python -c "import torch" &> /dev/null; then
    echo "⚠️  Instalando dependencias..."
    pip install -r requirements.txt
else
    echo "✅ PyTorch ya está instalado"
    # Verificar otras dependencias críticas
    if ! python -c "import transformers, datasets, peft, deepspeed" &> /dev/null; then
        echo "⚠️  Instalando dependencias faltantes..."
        pip install -r requirements.txt
    fi
fi

# Paso 3: Descargar modelo Mistral 7B Instruct si no existe
echo "📥 Verificando modelo Mistral 7B Instruct..."
if [ ! -d "~/.cache/huggingface/hub" ] || ! python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')" &> /dev/null; then
    echo "📥 Descargando modelo Mistral 7B Instruct..."
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Descargando tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
print('Descargando modelo...')
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', torch_dtype='auto')
print('✅ Modelo descargado exitosamente')
"
else
    echo "✅ Modelo Mistral ya está disponible"
fi

# Paso 4: Generar datos de entrenamiento desde synth
echo "🔄 Generando datos de entrenamiento..."
if [ -d "synth" ] && [ -f "synth/format.py" ]; then
    cd synth
    echo "📝 Ejecutando format.py..."
    python format.py
    
    if [ -f "formatted_data.jsonl" ]; then
        echo "✅ Datos formateados generados exitosamente"
        # Mover el archivo al directorio principal
        mv formatted_data.jsonl ../training_data.jsonl
        echo "📁 Archivo movido a training_data.jsonl"
    else
        echo "❌ Error: No se generó formatted_data.jsonl"
        cd ..
        exit 1
    fi
    cd ..
else
    echo "❌ Error: No se encontró la carpeta synth o format.py"
    exit 1
fi

# Paso 5: Verificar que el archivo de datos existe
if [ ! -f "training_data.jsonl" ]; then
    echo "❌ Error: No se encontró training_data.jsonl después de la generación"
    exit 1
fi

# Verificar formato del archivo
echo "🔍 Verificando formato de datos..."
first_line=$(head -n 1 training_data.jsonl)
if echo "$first_line" | grep -q '"text".*<s>\[INST\].*\[/INST\].*</s>'; then
    echo "✅ Formato de datos correcto"
else
    echo "⚠️  Advertencia: El formato de datos podría no ser el esperado"
    echo "Primera línea: $first_line"
fi

# Mostrar estadísticas del dataset
num_lines=$(wc -l < training_data.jsonl)
echo "📊 Dataset contiene $num_lines ejemplos de entrenamiento"

# Paso 6: Configurar variables de entorno para optimización
echo "⚙️  Configurando entorno para entrenamiento distribuido..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Verificar GPUs disponibles
echo "🔍 Verificando GPUs disponibles..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "✅ Detectadas $gpu_count GPUs"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi no disponible, continuando..."
fi

# Paso 7: Lanzar entrenamiento con DeepSpeed
echo "🚀 Iniciando entrenamiento distribuido..."
echo "📊 Configuración:"
echo "  - Modelo: Mistral 7B Instruct"
echo "  - GPUs: 8x H100"
echo "  - Batch size efectivo: 128"
echo "  - DeepSpeed ZeRO Stage 2"
echo "  - Precision: BFloat16"
echo ""

# Crear directorio de logs si no existe
mkdir -p logs

# Lanzar entrenamiento con logging
deepspeed --num_gpus=8 \
    --master_port=29500 \
    main.py \
    --deepspeed ds_config.json 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Verificar si el entrenamiento fue exitoso
if [ $? -eq 0 ]; then
    echo "✅ Entrenamiento completado exitosamente!"
    echo "📁 Modelo guardado en: ./mistral_7b_frida_finetuned"
    echo "📊 Logs guardados en: logs/"
    
    # Mostrar información del modelo entrenado
    if [ -d "./mistral_7b_frida_finetuned" ]; then
        model_size=$(du -sh ./mistral_7b_frida_finetuned | cut -f1)
        echo "💾 Tamaño del modelo: $model_size"
    fi
else
    echo "❌ Error durante el entrenamiento"
    echo "📋 Revisa los logs en: logs/"
    exit 1
fi
