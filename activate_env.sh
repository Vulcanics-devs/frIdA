#!/bin/bash

# Script para activar el entorno virtual de frIdA
# Uso: source activate_env.sh

echo "🐍 Activando entorno virtual frIdA..."

# Verificar si el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Error: No se encontró el entorno virtual"
    echo "💡 Ejecuta primero: bash launch_training.sh"
    echo "   O crea manualmente: python -m venv venv"
    return 1 2>/dev/null || exit 1
fi

# Activar entorno virtual
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✅ Entorno virtual activado"
    echo "🐍 Python: $(which python)"
    echo "📦 Pip: $(which pip)"
    
    # Mostrar versiones importantes
    echo ""
    echo "📊 Versiones instaladas:"
    python -c "
import sys
print(f'Python: {sys.version.split()[0]}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA disponible: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPUs detectadas: {torch.cuda.device_count()}')
except ImportError:
    print('PyTorch: No instalado')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('Transformers: No instalado')

try:
    import deepspeed
    print(f'DeepSpeed: {deepspeed.__version__}')
except ImportError:
    print('DeepSpeed: No instalado')
"
else
    echo "❌ Error: No se pudo activar el entorno virtual"
    return 1 2>/dev/null || exit 1
fi
