# Polish MLX Tools

Narzędzia do pracy z polskimi modelami językowymi na Apple Silicon przy użyciu MLX.

## Zawartość

- `scripts/convert_simple.py` - skrypt do konwersji modeli z formatu HF/GGUF do MLX
- `scripts/aliases.zsh` - aliasy do shell'a ułatwiające pracę z MLX
- `scripts/prepare_finetuning.py` - skrypt do przygotowania danych do fine-tuningu

## Wspierane modele

- Bielik (11B)
- PLLuM (12B)
- Eskulap Alpha 1

## Szybki start

```bash
# Instalacja środowiska
uv venv
source .venv/bin/activate
uv pip install mlx mlx-lm huggingface_hub hf_transfer

# Pobieranie modelu
mkdir -p models/bielik
cd models/bielik
huggingface-cli download speakleash/Bielik-11B-v2.3-Instruct --local-dir ./
cd ../..

# Konwersja do MLX
python3 scripts/convert_simple.py --input-dir models/bielik --output-dir models/Bielik-MLX --force

# Kwantyzacja 8-bit
python3 scripts/convert_simple.py --input-dir models/bielik --output-dir models/Bielik-MLX-8bit --quantize 8 --q-group-size 128 --force

# Generacja tekstu
python3 -m mlx_lm.generate --model models/Bielik-MLX-8bit --prompt "Wyjaśnij, jak działa rekurencja w programowaniu." --max-tokens 200
```

## Aliasy

Dodaj do .zshrc lub .bashrc:

```bash
source /pełna/ścieżka/do/scripts/aliases.zsh
```

## Licencja

MIT