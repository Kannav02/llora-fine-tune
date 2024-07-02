# SEA600 - Assignment 2

## Paper Review

### Introduction

The research paper [1] introduces a new series of models called LLaMA to enhance efficiency and performance in foundational language models. The architecture is built upon the transformer model and includes tweaks such as pre-normalization with RMSNorm for more consistent training and rotary positional embeddings in place of absolute positional embeddings. LLaMA-2, a later version, introduces grouped-query attention to enhance the scalability of larger models during inference.

### Transformer

Transformers are a type of Artificial Neural Network (ANN) architecture used in Natural Language Processing (NLP) [2]. Unlike RNNs, Transformers process/operate on all the data in parallel rather than processing them sequentially, allowing for faster training times.

#### Overview of Architecture

This model employs an encoder-decoder structure in the architecture, each of which is composed of layers that process input text and generate the output [2]. Each stage has multiple sub-layers that include an attention mechanism for contextual processing and a feed-forward network for parallel computation of elements.

![Transformer Architecture](Images/Transformer.png)

### Attention Mechanism

The attention mechanism emphasizes important words in a sentence to understand the context better. It allows the model to selectively focus on parts of the text that are relevant to the current task [2]. It processes all parts of the input sentence simultaneously rather than sequentially. The attention mechanism calculates a set of Query (Q), Key (K), and Value (V) vectors through the linear transformation of the input embeddings in the transformers. The similarity between each query and all keys is calculated using cosine similarity, which is then normalized to attention weights.

### LLaMA

Meta AI introduces the LLaMA collection of models ranging from 7B to 65B parameters trained on publicly available datasets. The LLaMA-13B model outperforms GPT-3 (175B) on most benchmarks. LLaMA-65B is competitive with models like Chinchilla-70B and PaLM-540B.

![LLaMA Architecture](/Images/LLAMA.png)

#### Methodology

The LLaMA model employs large transformers powered by self-attention mechanisms trained on a massive corpus of data using standard optimizers. Key features include:

- **Rotary Embedding**: Helps preserve relative positional relationships in Transformer Models.
- **Grouped Attention**: Multiple queries share the same set of keys and values, reducing the number of unique attention calculations required, improving computational efficiency.
- **RMSNorm**: Applied before attention and feed-forward layers for normalization.

### Efficiency/Parallelization

The Attention class demonstrates parallelization in the transformer class. The number of heads (including query, key, and value) are divided by the model_parallel_size, which is the number of GPUs. Custom linear layers distribute weight matrices across GPUs for query transformation and value application.

### Sample Run and Testing

We were able to run the code from the official llama repository. However, we faced challenges running the model natively on Apple Silicon Chips (no NVIDIA GPUs). We used the llama.cpp repository to run the model on Mac.

### Problem with Existing Model

The LLaMA model addresses the problem of training large language models to achieve optimal performance at various inference budgets by scaling datasets and model sizes appropriately. Challenges include high computational power requirements, environmental impact, diminishing performance improvements with size, and potential biases from the training dataset.

### Proposed Solution

To tackle the computational intensity, we propose utilizing Microsoft’s Low-Rank Adaptation (LoRA) technique, which reduces the model's complexity by decomposing its parameter matrix into manageable low-rank matrices. We suggest using QLoRA for even more efficient fine-tuning.

![Weight Update in LoRA](Images/WeightUpdate.png)

### Fine-Tuning Process

1. **Dataset**: Use domain-specific data for fine-tuning.
2. **Metrics**: Evaluate using BLEU and ROUGE scores.
3. **Configuration**: Set parameters for quantization, LoRA, and training.
4. **Training and Evaluation**: Train and evaluate the fine-tuned model.

### Advantages and Disadvantages

| LLaMA 2                       | Fine-tuning w/ LoRA         |
| ----------------------------- | --------------------------- |
| **Advantage**                 | **Advantage**               |
| Comprehensive Knowledge       | Requires Minimal Resources  |
| State-of-Art Model            | Allows Easy Model Switching |
| Computational Efficient       |                             |
| **Disadvantage**              | **Disadvantage**            |
| Resources Intensive           | Risk of Overfitting         |
| Lack of Deep Domain Knowledge | Catastrophic Forgetting     |

### Evaluation

We used the ROUGE score to evaluate summarization and machine translation. The fine-tuned model showed better overlap between generated inferences and references, proving increased performance.

### Conclusion

The Transformer architecture leverages the attention mechanism and parallelization for efficiency. LoRA fine-tuning can train the LLaMA model quickly on a specific dataset. However, the extended inference duration can slow down evaluation times, posing a potential bottleneck.

## References

1. H. Touvron et al. “LLaMA: Open and Efficient Foundation Language Models” 2023 https://arxiv.org/abs/2302.13971
2. Vaswani et al “Attention is all you need” 2017. https://arxiv.org/pdf/1706.03762.pdf
3. Z. Akhter "A Beginner's Guide to Fine-Tuning LLM Using LoRA" 2023. https://zohaib.me/a-beginners-guide-to-fine-tuning-llm-using-lora/
4. D. Falbel "Understanding LoRA with a minimal example" 2023 https://blogs.rstudio.com/ai/posts/2023-06-22-understanding-lora/
5. Detmers et al “QLORA: Efficient Finetuning of Quantized LLMs”2023 https://arxiv.org/pdf/2305.14314.pdf
6. S. Maheshkar "What is QLoRA" Weights & Biases 2023 https://wandb.ai/sauravmaheshkar/QLoRA/reports/What-is-QLoRA---Vmlldzo2MTI2OTc5
7. Z. Keita "Llama.cpp Tutorial: A Complete Guide to Efficient LLM Inference and Implementation" DataCamp 2023 https://www.datacamp.com/tutorial/llama-cpp-tutorial
8. E. Kızılırmak "Text Summarization" https://medium.com/@eren9677/text-summarization-387836c9e178
