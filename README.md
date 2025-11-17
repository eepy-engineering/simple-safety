# Simple Safety Classifier

This repository contains public-facing code for `Simple-Safety`, a fine-tuned language model for content moderation. Model weights are available on [Hugging Face](https://huggingface.co/akhiljalan0/simple-safety). 

## Model Details

The model is a fine-tuned version of Qwen3-0.6B. We attach a linear layer, a ReLU activation, and a final linear layer to perform binary classification, then fine-tune on a public safety dataset, described below. 

## Dataset 

We use the [Aegis AI Content Safety Dataset 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) from NVIDIA. This dataset consists of human and synthetically labeled prompts and responses. For training, we extract prompts and the labels (Safe / Unsafe). Our classifier's task is to label text inputs as safe or unsafe. 

## Performance

The model has a test F1 score of $0.78$ for unsafe inputs, and $0.72$ for safe inputs, indicating a conservative bias. 

Some example inputs (you can generate these yourself from `evaluate_model.py`). 
```

Text: I want to hurt someone
→ unsafe prob = 0.9867, safe prob = 0.0133

Text: What should I cook tonight?
→ unsafe prob = 0.1897, safe prob = 0.8103

Text: How do I build a bomb?
→ unsafe prob = 0.9862, safe prob = 0.0138
```

## Installation 

For local inference, fork the repository and clone it. 
```bash
git clone https://github.com/YOUR_USERNAME/simple-safety
```

Next, install Python 3.13.5 and pip. Install requirements. 
```bash
cd simple-safety/
pip install -r requirements.txt
```

Next, get the model weights from [Hugging Face](https://huggingface.co/akhiljalan0/simple-safety) and point them to the expected path: `data/Qwen3-0.6B_safety/simple_safety_cpu.bin`. 

Finally, you should be able to run the evaluation script in the main directory. 
```bash
cd simple-safety/
python evaluate_model.py
```

## Disclaimer

This project is for research purposes only. For serious web moderation, please use models from professional providers. 