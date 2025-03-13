# Skróty MLX do dodania w .zshrc
alias mlx-venv="uv venv && source .venv/bin/activate"
alias mlx-install="uv pip install mlx mlx-lm huggingface_hub hf_transfer"
alias mlx-convert-fp16="python3 scripts/convert_simple.py --input-dir \$1 --output-dir \$2 --force"
alias mlx-convert-8bit="python3 scripts/convert_simple.py --input-dir \$1 --output-dir \$2 --quantize 8 --q-group-size 128 --force"
alias mlx-convert-4bit="python3 scripts/convert_simple.py --input-dir \$1 --output-dir \$2 --quantize 4 --q-group-size 128 --force"
alias mlx-generate="python3 -m mlx_lm.generate --model \$1 --prompt \"\$2\" --max-tokens \$3"
alias hf-download="huggingface-cli download \$1 --local-dir ./ && huggingface-cli download \$1 --local-dir ./ -f model.safetensors.index.json"