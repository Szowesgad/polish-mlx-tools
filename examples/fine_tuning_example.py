#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Przykładowy skrypt do fine-tuningu modelu MLX używając LoRA/QLoRA.
"""

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning modelu MLX")
    parser.add_argument("--model", required=True, help="Ścieżka do modelu MLX")
    parser.add_argument("--data-dir", required=True, help="Katalog z plikami train.jsonl i valid.jsonl")
    parser.add_argument("--output-adapter", default="adapters/adapter", help="Ścieżka do zapisania adaptera")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Długość sekwencji")
    parser.add_argument("--iters", type=int, default=1000, help="Liczba iteracji treningu")
    parser.add_argument("--lora-rank", type=int, default=32, help="Rank LoRA")
    parser.add_argument("--lora-alpha", type=int, default=64, help="Alpha LoRA")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout LoRA")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="Weight decay")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Użyj gradient checkpointing")
    args = parser.parse_args()
    
    # Upewnij się, że katalog na adapter istnieje
    os.makedirs(os.path.dirname(args.output_adapter), exist_ok=True)
    
    # Przygotuj polecenie treningu
    cmd = [
        "python3", "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train", "--data", args.data_dir,
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--adapter-path", args.output_adapter
    ]
    
    # Dodaj gradient checkpointing, jeśli jest włączony
    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    
    # Uruchom trening
    print(f"Uruchamianie treningu z komendą:\n{' '.join(cmd)}")
    subprocess.run(cmd)
    
    # Pokaż informacje po treningu
    print(f"\n✅ Trening zakończony.")
    print(f"Adapter został zapisany w: {args.output_adapter}")
    print("\nAby użyć adaptera do generacji, użyj:")
    print(f"python3 -m mlx_lm.generate --model {args.model} --adapter-path {args.output_adapter} --prompt \"Twój prompt\" --max-tokens 200")
    print("\nAby zfuzować adapter z modelem bazowym:")
    print(f"python3 /ścieżka/do/fuse.py --model {args.model} --adapter-file {args.output_adapter}/adapter.safetensors --save-path {args.model}-Fused")

if __name__ == "__main__":
    main()