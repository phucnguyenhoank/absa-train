from transformers import AutoTokenizer, AutoModel

backbone_model_name = "vinai/phobert-base-v2"
save_directory = "./phobert-base-v2"

# 1. Load from Hub
tokenizer = AutoTokenizer.from_pretrained(backbone_model_name)
model = AutoModel.from_pretrained(backbone_model_name)

# 2. Save to local folder
# This creates files like config.json, pytorch_model.bin, and vocab.txt
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

# 3. Reload from local folder
# Simply point the 'from_pretrained' path to your local directory
local_tokenizer = AutoTokenizer.from_pretrained(save_directory)
local_model = AutoModel.from_pretrained(save_directory)
