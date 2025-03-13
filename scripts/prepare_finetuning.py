#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt do przygotowywania danych do fine-tuningu modeli w MLX.
"""

import argparse
import json
import os
import random

def convert_to_instruction_format(input_data, output_format="default"):
    """
    Konwertuje dane wejściowe do formatu instrukcyjnego dla MLX.
    
    Formaty:
    - default: {"prompt": "...", "completion": "..."}
    - chat: {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
    """
    results = []
    
    for item in input_data:
        if output_format == "chat":
            formatted_item = {
                "text": f"<|im_start|>user\n{item['instruction']}\n<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
            }
        else:  # default
            prompt = f"Zadanie: {item['instruction']}\n"
            if 'input' in item and item['input']:
                prompt += f"{item['input']}\n"
            prompt += "Odpowiedź:"
            
            formatted_item = {
                "prompt": prompt,
                "completion": item['output']
            }
        
        results.append(formatted_item)
    
    return results

def save_jsonl(data, filename):
    """Zapisuje dane w formacie JSONL."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Przygotowanie danych do fine-tuningu MLX")
    parser.add_argument("--input", required=True, help="Ścieżka do pliku wejściowego JSON")
    parser.add_argument("--output-dir", default="data", help="Katalog wyjściowy dla plików JSONL")
    parser.add_argument("--format", choices=["default", "chat"], default="default", 
                      help="Format wyjściowy (default: prompt/completion, chat: text z tokenami chat)")
    parser.add_argument("--split", type=float, default=0.9, 
                      help="Proporcja podziału train/valid (np. 0.9 oznacza 90% train, 10% valid)")
    
    args = parser.parse_args()
    
    # Wczytaj dane
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Konwersja formatu
    formatted_data = convert_to_instruction_format(data, args.format)
    
    # Podział na zbiory treningowy i walidacyjny
    random.shuffle(formatted_data)
    split_idx = int(len(formatted_data) * args.split)
    train_data = formatted_data[:split_idx]
    valid_data = formatted_data[split_idx:]
    
    # Utwórz katalog wyjściowy, jeśli nie istnieje
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Zapisz dane
    save_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(valid_data, os.path.join(args.output_dir, "valid.jsonl"))
    
    print(f"✅ Zapisano {len(train_data)} przykładów w train.jsonl i {len(valid_data)} w valid.jsonl")

if __name__ == "__main__":
    main()