#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Konwerter modeli HF do MLX")
    parser.add_argument("--input-dir", required=True, help="Katalog wejściowy modelu")
    parser.add_argument("--output-dir", required=True, help="Katalog wyjściowy modelu MLX")
    parser.add_argument("--quantize", type=int, choices=[4, 8], help="Kwantyzacja (4 lub 8 bitów)")
    parser.add_argument("--q-group-size", type=int, help="Rozmiar grupy kwantyzacji (np. 128)")
    parser.add_argument("--force", action="store_true", help="Usuń katalog docelowy jeśli istnieje")
    parser.add_argument("--verbose", action="store_true", help="Tryb debugowania")
    args = parser.parse_args()
    
    # Usuń katalog docelowy jeśli istnieje i podano --force
    if os.path.exists(args.output_dir) and args.force:
        print(f"🗑️ Usuwam katalog docelowy: {args.output_dir}")
        try:
            # Usuń katalog za pomocą polecenia systemowego (bardziej niezawodne)
            subprocess.run(['rm', '-rf', args.output_dir], check=True)
            print("✅ Katalog usunięty")
        except Exception as e:
            print(f"❌ Błąd podczas usuwania katalogu: {e}")
            return
    
    # Przygotuj polecenie konwersji
    cmd = [sys.executable, "-m", "mlx_lm.convert", 
           "--hf-path", args.input_dir, 
           "--mlx-path", args.output_dir]
    
    # Dodaj opcje kwantyzacji
    if args.quantize:
        cmd.append("-q")
        if args.quantize != 4:  # 4 bity są domyślne
            cmd.extend(["--q-bits", str(args.quantize)])
        if args.q_group_size:
            cmd.extend(["--q-group-size", str(args.q_group_size)])
    
    # Dodaj tryb verbose
    if args.verbose:
        cmd.append("-d")
    
    print(f"Wykonuję: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()