# Fine-Tuning Llama 2 for Financial Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

A project focused on fine-tuning the Llama 2 7B model for sentiment analysis on financial news headlines using QLoRA (Quantized Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning) techniques.

## Overview

This project implements a fine-tuned sentiment classifier that categorizes financial news headlines into three classes: positive, neutral, and negative. The model achieves significant performance improvements over the base Llama 2 model by leveraging efficient fine-tuning methods that reduce computational requirements while maintaining high accuracy.

## Dataset

The project uses the FinancialPhraseBank dataset, which contains approximately 5,000 financial news headlines annotated by 16 domain experts. Each headline is labeled based on its potential impact on stock prices from a retail investor's perspective.

- **Classes**: Positive, Neutral, Negative
- **Training samples**: 900 (300 per class)
- **Test samples**: 900 (300 per class)

## Key Features

- 4-bit quantization using bitsandbytes for memory-efficient model loading
- LoRA-based fine-tuning with rank 64 and alpha 16
- Gradient checkpointing for reduced memory footprint
- Cosine learning rate scheduler with warmup
- TensorBoard integration for training visualization

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- PyTorch 2.1.2
- Transformers 4.36.2
- PEFT
- TRL (Transformer Reinforcement Learning)
- bitsandbytes 0.42.0
- accelerate 0.26.1

## Installation

```bash
pip install torch==2.1.2 tensorboard
pip install transformers==4.36.2 datasets==2.16.1 accelerate==0.26.1 bitsandbytes==0.42.0
pip install git+https://github.com/huggingface/trl
pip install git+https://github.com/huggingface/peft
```

## Project Structure

```
fine-tune-llama-2-for-sentiment-analysis/
|-- fine-tune-llama-2-for-sentiment-analysis.ipynb
|-- README.md
|-- trained_weights/          # Fine-tuned adapter weights
|-- merged_model/             # Merged model for inference
|-- test_predictions.csv      # Evaluation results
```

## Training Configuration

| Parameter             | Value         |
| --------------------- | ------------- |
| Base Model            | Llama 2 7B HF |
| Quantization          | 4-bit NF4     |
| LoRA Rank             | 64            |
| LoRA Alpha            | 16            |
| Learning Rate         | 2e-4          |
| Epochs                | 3             |
| Batch Size            | 1             |
| Gradient Accumulation | 8             |
| Max Sequence Length   | 1024          |

## Results

The fine-tuned model demonstrates substantial improvement over the base model:

| Metric                  | Base Model | Fine-Tuned Model |
| ----------------------- | ---------- | ---------------- |
| Overall Accuracy        | ~33%       | 80%+             |
| Positive Class Accuracy | Low        | High             |
| Negative Class Accuracy | Low        | High             |
| Neutral Class Accuracy  | Low        | Moderate         |

## Usage

1. Load the fine-tuned model:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("./trained_weights/")
tokenizer = AutoTokenizer.from_pretrained("./merged_model/")
```

2. Run inference:

```python
prompt = """Analyze the sentiment of the news headline enclosed in square brackets,
determine if it is positive, neutral, or negative, and return the answer as
the corresponding sentiment label "positive" or "neutral" or "negative".

[Company reports record quarterly earnings] = """

# Generate prediction using the model
```

## Acknowledgments

- Meta AI for the Llama 2 model
- Hugging Face for the transformers and PEFT libraries
- Aalto University School of Business for the FinancialPhraseBank dataset

## References

- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P. (2014). "Good debt or bad debt: Detecting semantic orientations in economic texts." Journal of the Association for Information Science and Technology.

## License

This project is licensed under the MIT License.
