# Hateful Illlusions

This repository contains the official implementation of the paper **Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions** (ICCV 2025).
[(arXiv link)](https://arxiv.org/abs/2507.22617)
---

## 📦 Installation

Prerequisite: Python 3.9+; CUDA

### 1. Basic environment

```bash
bash setup.sh
```

### 2. Customized environments for different VLMs (See details in the Detection part)

See:

```
scripts/run_xxx.sh
```

### 3. Tokens (See details in the Detection part)

This repo requires API keys from **OpenAI**, **Google**, and **Microsoft Azure**.  
Please set the keys inside the corresponding files under `scripts/run_xxx.sh`.

---

## 🎯 Hateful Illusion Dataset

The dataset contains **2,160 AI-generated (hateful) optical illusions**, each hiding one of three message types:

| Category | Messages | Images |
|----------|----------|--------|
| digits | 10 | 300 |
| hate slangs (hate speech) | 23 | 690 |
| hate symbols | 39 | 1,170 |

Each image is annotated with a visibility level:

- **0 – No visibility:** hidden message is not visible  
- **1 – Low visibility:** subtly visible  
- **2 – High visibility:** clearly visible  

Download from HuggingFace:  
https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset

---

### How to use the dataset

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download

repo_id = "yiting/HatefulIllusion_Dataset"
local_dir = "data/HatefulIllusion_Dataset"  # example path

snapshot_download(
    repo_id,
    repo_type="dataset",
    local_dir=local_dir
)

subset = "digits"
# subset = "hate_slangs"
# subset = "hate_symbols"

dataset = load_dataset(local_dir, subset)["train"]
print(dataset[0])
```

Example output:

```python
{
  "image": "images/illusion_000123.png",
  "message": "embedded hate slang meaning X",
  "condition_image": "messages/cond_000123.png",
  "prompt": "Generate an optical illusion containing ...",
  "visibility": 1
}
```

The dataset is automatically downloaded to:

```
data/HatefulIllusion_Dataset
```

when running `detect.py`.

---

## 🎨 Dataset Generation

Generate optical illusions:

```bash
python generate_illusions.py --target "digits" --image_root "data" --save_root "data/generations"
python generate_illusions.py --target "hate_slangs" --image_root "data" --save_root "data/generations"
python generate_illusions.py --target "hate_symbols" --image_root "data" --save_root "data/generations"
```

Valid `--target` values:

- digits  
- hate_slangs (corresponding to hate_speech in our paper)  
- hate_symbols  

---

## 🔍 Detection

Evaluate detection performance of VLMs and image safety moderators:

```bash
python $DIR/detect.py \
    --model_name azure \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target \
    --api_key your_api_key \
    --endpoint your_project_endpoint
```

### Arguments

| Argument | Description |
|----------|-------------|
| model_name | VLM or detector name |
| dataset_name | `"illusion"` (illusion image) or `"message"` (message image) |
| query_mode | `"zero-shot"` or `"cot"` |
| target | digits, hate_slangs, hate_symbols |
| api_key | Required for commercial VLMs/moderators |
| endpoint | Required for Azure moderators |

---

## 🧠 Vision-Language Models (VLMs)

### GPT series (GPT-4V, GPT-4o)

```bash
bash scripts/run_gpt.sh
```

### Gemini series (Gemini-1.5, Gemini-2)

```bash
bash scripts/run_gemini.sh
```

### LLaVA series (LLaVA-1.5, LLaVA-Next)

```bash
bash scripts/run_llava.sh
```

### Qwen-VL

```bash
bash scripts/run_qwen.sh
```

### CogVLM series

```bash
bash scripts/run_cogvlm.sh
```

---

## 🛡️ Image Safety Moderators

### Q16 and Stable Diffusion Safety Checker

```bash
bash scripts/run_q16.sh
bash scripts/run_safety_checker.sh
```

### Google SafeSearch

1. Download Google Cloud credentials: https://console.cloud.google.com  
2. Save as:

```
.env/google_cloud.json
```

3. Follow instructions: https://docs.cloud.google.com/iam/docs/keys-create-delete

Run:

```bash
bash scripts/run_safesearch.sh
```

### Microsoft Azure Moderators

Setup API key and endpoint via https://portal.azure.com.

```bash
bash scripts/run_azure.sh
```

---

## 📚 Citation

```bibtex
@inproceedings{QYMBZ25,
  author    = {Yiting Qu and Ziqing Yang and Yihan Ma and Michael Backes and Yang Zhang},
  title     = {Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  publisher = {ICCV},
  year      = {2025}
}
```

