# 🔥 Fine-tuning Mistral 7B Instruct - frIdA

Script optimizado para fine-tuning de Mistral 7B Instruct con 8 GPUs H100 usando DeepSpeed y LoRA.

## 🚀 Características

- **Modelo**: Mistral 7B Instruct v0.2
- **Hardware**: Optimizado para 8x H100 GPUs
- **Técnicas**: LoRA + DeepSpeed ZeRO Stage 2
- **Precisión**: BFloat16 para máximo rendimiento
- **Batch Size Efectivo**: 128 (8 GPUs × 8 batch × 2 accumulation)

## 📋 Requisitos

- Python 3.8+
- CUDA 11.8+
- 8x H100 GPUs (mínimo 80GB VRAM cada una)
- ~200GB espacio libre en disco

## 🛠️ Instalación

1. **Clonar el repositorio**:
```bash
git clone <tu-repo>
cd frida
```

2. **Crear entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate     # En Windows
```

3. **Instalar dependencias**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota**: El script `launch_training.sh` crea automáticamente el entorno virtual si no existe.

## 📊 Estructura de Datos

El script espera datos en formato JSONL con la estructura de Mistral:

```json
{"text": "<s>[INST] pregunta del usuario [/INST] respuesta del asistente </s>"}
```

## 🚀 Uso

### Entrenamiento Automático

El script `launch_training.sh` automatiza todo el proceso:

```bash
bash launch_training.sh
```

Este script:
1. 🐍 Crea y activa entorno virtual automáticamente
2. ✅ Verifica e instala dependencias
3. 📥 Descarga el modelo Mistral 7B Instruct
4. 🔄 Genera datos de entrenamiento desde `synth/format.py`
5. 🚀 Inicia el entrenamiento distribuido
6. 💾 Guarda el modelo entrenado

### Entrenamiento Manual

Si prefieres ejecutar manualmente:

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generar datos
cd synth
python format.py
mv formatted_data.jsonl ../training_data.jsonl
cd ..

# 3. Entrenar (básico)
deepspeed --num_gpus=8 main.py --deepspeed ds_config.json

# 3. Entrenar (con opciones personalizadas)
deepspeed --num_gpus=8 main.py --deepspeed ds_config.json \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --data mi_dataset.jsonl \
  --output ./mi_modelo_entrenado
```

### Opciones de Línea de Comandos

| Opción | Predeterminado | Descripción |
|--------|----------------|-------------|
| `--model` | `mistralai/Mistral-7B-Instruct-v0.2` | Modelo base a usar |
| `--data` | `training_data.jsonl` | Archivo de datos de entrenamiento |
| `--output` | `./mistral_7b_frida_finetuned` | Directorio de salida |
| `--epochs` | Auto-calculado | Número de épocas (sobrescribe cálculo) |
| `--batch-size` | `8` | Batch size por dispositivo |
| `--learning-rate` | `1e-4` | Learning rate |

### Lógica de Épocas Automática

El script calcula automáticamente el número óptimo de épocas según el tamaño del dataset:

| Tamaño del Dataset | Épocas | Razón |
|---------------------|--------|--------|
| < 1,000 ejemplos | 5 | Dataset pequeño, necesita más pasadas |
| 1,000 - 10,000 | 4 | Dataset pequeño-mediano |
| 10,000 - 50,000 | 3 | Dataset mediano (óptimo) |
| 50,000 - 100,000 | 2 | Dataset grande |
| > 100,000 | 1 | Dataset muy grande, evita overfitting |

> **Tip**: Puedes sobrescribir con `--epochs N` si tienes necesidades específicas.

### Probar el Modelo

Después del entrenamiento, puedes probar el modelo:

```bash
# Activar entorno virtual (si no está activo)
source venv/bin/activate

# Pruebas predefinidas + modo interactivo
python test_model.py

# Solo modo interactivo
python test_model.py --interactive

# Solo pruebas predefinidas
python test_model.py --batch
```

## 🐍 Gestión del Entorno Virtual

### Activación Rápida

Para activar el entorno virtual manualmente:

```bash
# Activar con información detallada
source activate_env.sh

# O activación simple
source venv/bin/activate
```

### Verificar Instalación

Después de activar el entorno:

```bash
# Verificar versiones
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"

# Verificar GPUs
nvidia-smi
```

### Desactivar Entorno

```bash
deactivate
```

### Recrear Entorno (si hay problemas)

```bash
# Eliminar entorno existente
rm -rf venv

# Crear nuevo entorno
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## ⚙️ Configuración

### DeepSpeed (ds_config.json)

- **ZeRO Stage 2**: Optimización de memoria distribuida
- **BFloat16**: Precisión optimizada para H100
- **Gradient Clipping**: Estabilidad de entrenamiento
- **AdamW Optimizer**: Con cosine learning rate schedule

### LoRA Configuration

- **Rank**: 64 (alta calidad)
- **Alpha**: 128
- **Target Modules**: Todas las proyecciones de atención y MLP
- **Dropout**: 0.05

### Training Arguments

- **Epochs**: 3
- **Learning Rate**: 1e-4 con warmup
- **Batch Size**: 8 por GPU
- **Gradient Accumulation**: 2 steps
- **Max Length**: 2048 tokens

## 📁 Estructura del Proyecto

```
frida/
├── main.py                 # Script principal de entrenamiento
├── launch_training.sh      # Script automatizado de lanzamiento
├── activate_env.sh         # Script para activar entorno virtual
├── test_model.py          # Script para probar el modelo
├── ds_config.json         # Configuración DeepSpeed
├── requirements.txt       # Dependencias Python
├── README.md              # Documentación
├── venv/                  # Entorno virtual (creado automáticamente)
├── synth/                 # Carpeta con datos sintéticos
│   ├── format.py         # Script para formatear datos
│   └── ...
├── logs/                  # Logs de entrenamiento
├── training_data.jsonl    # Datos de entrenamiento (generado)
└── mistral_7b_frida_finetuned/  # Modelo entrenado
```

## 🔧 Optimizaciones

### Para H100 GPUs:
- Flash Attention 2
- BFloat16 precision
- TensorFloat-32 (TF32)
- Optimized memory allocation
- NCCL optimizations

### Para Memoria:
- Gradient checkpointing
- DeepSpeed ZeRO Stage 2
- Dynamic padding
- LoRA (Low-Rank Adaptation)

## 📊 Monitoreo

- **TensorBoard**: `tensorboard --logdir logs/`
- **Logs**: Guardados en `logs/training_YYYYMMDD_HHMMSS.log`
- **GPU Usage**: `nvidia-smi` durante entrenamiento

## 🐛 Troubleshooting

### Error de Memoria GPU
```bash
# Reducir batch size en main.py
per_device_train_batch_size=4  # En lugar de 8
```

### Error de Conectividad Multi-GPU
```bash
# Verificar NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

### Modelo no Descarga
```bash
# Descargar manualmente
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2'); AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
```

## 📈 Resultados Esperados

Con 8x H100 GPUs:
- **Velocidad**: ~2-3 horas para 3 epochs (depende del dataset)
- **Memoria**: ~60-70GB por GPU
- **Throughput**: ~1000-1500 tokens/segundo

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

---

**¡Feliz fine-tuning! 🚀**
