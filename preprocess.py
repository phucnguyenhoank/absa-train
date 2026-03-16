# This contains segmenter and tokenizer for the raw input text

from transformers import AutoTokenizer
from pathlib import Path
import py_vncorenlp
import os

from config import backbone_model_name

tokenizer = AutoTokenizer.from_pretrained(backbone_model_name)

cwd = Path.cwd()

# Define the ABSOLUTE path
save_dir = (cwd / "vncorenlp").resolve()

# Manually create the parent directory first
# The library's download_model fails because it can't create /vncorenlp/models
# if /vncorenlp doesn't exist yet.
save_dir.mkdir(parents=True, exist_ok=True)

if not (save_dir / "models").exists():
    print(f"Downloading models to {save_dir}...")
    # The command below tried to run a command like: Create folder "models" inside "vncorenlp"
    # BUT it does not create the "vncorenlp" folder.
    # VERY VERY FRUSTRATING
    py_vncorenlp.download_model(save_dir=save_dir.as_posix())
else:
    print("Models already exist, skipping download.")


try:
    rdrsegmenter = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"], save_dir=save_dir.as_posix()
    )
    print("Segmenter loaded successfully!")
except Exception as e:
    print(f"Error loading segmenter: {e}")

os.chdir(cwd)
print(Path.cwd())
