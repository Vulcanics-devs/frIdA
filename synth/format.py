import json

input_path = "synthetic_data.jsonl"
output_path = "formatted_data.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        entry = json.loads(line)
        prompt = entry.get("prompt", "").strip()

        # Elimina los delimitadores si ya vienen incluidos
        prompt = prompt.replace("<s>", "").replace("</s>", "").strip()
        if prompt.startswith("[INST]") and prompt.endswith("[/INST]"):
            prompt = prompt[len("[INST]"): -len("[/INST]")].strip()

        response = entry.get("response", "").strip()
        text = f"<s>[INST] {prompt} [/INST] {response}</s>"
        json.dump({"text": text}, outfile, ensure_ascii=False)
        outfile.write("\n")
