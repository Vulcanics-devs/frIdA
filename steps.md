Perfecto, Miltron. Aquí van las **instrucciones paso a paso** para bajar un modelo tipo **Mistral de OpenWeights**, y hacer **fine-tuning supervisado** usando un archivo `.jsonl`. Asumiré que tienes **infraestructura con GPU NVIDIA**, posiblemente con CUDA y entorno Linux.

---

## 🧱 Paso 0: Requisitos previos

Antes de comenzar, asegúrate de tener:

* Python ≥ 3.10
* CUDA ≥ 11.7 con `nvidia-smi` funcional
* Un entorno virtual (`venv` o `conda`)
* Archivo `.jsonl` con formato:

  ```json
  {"text": "Texto de entrenamiento aquí"}
  ```

---

## 🧰 Paso 1: Instalar dependencias

```bash
python3 -m venv mistral-ft
source mistral-ft/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft bitsandbytes
```

---

## 📦 Paso 2: Descargar el modelo Mistral de OpenWeights

Por ejemplo, si quieres usar **Mistral-7B-Instruct**:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)
```

O si estás en bash:

```bash
huggingface-cli login  # (si no lo has hecho antes)

# Clona el modelo a disco (opcional, si lo vas a usar offline)
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
```

---

## 🧹 Paso 3: Prepara el dataset `.jsonl`

Si tienes un archivo llamado `train.jsonl`, conviértelo a formato `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl")
```

Divide en `train/test` si lo necesitas:

```python
dataset = dataset["train"].train_test_split(test_size=0.05)
```

---

## 🧪 Paso 4: Configurar el trainer (QLoRA-style tuning)

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./mistral-ft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    save_steps=100,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator
)

trainer.train()
```

---

## ✅ Paso 5: Guardar y usar tu modelo fine-tuneado

```python
trainer.save_model("./mistral-finetuned")
tokenizer.save_pretrained("./mistral-finetuned")
```

Para usarlo:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="./mistral-finetuned", tokenizer=tokenizer, device=0)
pipe("Pregunta: ¿Qué opinas de México?\nRespuesta:")
```

---

¿Quieres usar **DeepSpeed**, **multi-GPU**, o directamente pasar a `FlashAttention` y `trt_llm` para inferencia extrema? Te doy la versión heavy si me dices.


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