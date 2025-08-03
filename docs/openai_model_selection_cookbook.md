# Practical Guide for Model Selection

This document summarizes insights from the OpenAI cookbook notebook ["Practical Guide for Model Selection for Real-World Use Cases"](https://github.com/openai/openai-cookbook/blob/main/examples/partners/model_selection_guide/model_selection_guide.ipynb) and relates them to the Knowledge3D (K3D) project.

> "This cookbook serves as your practical guide to selecting, prompting, and deploying the right OpenAI model..."

## Insights for K3D

1. **Decision frameworks** – Apply the guide's matrices to choose between GPT-4.1, o3, and o4-mini for tasks like vector generation, spatial reasoning, or AR narration.
2. **Template prompts** – Reuse the provided prompting patterns to standardize how K3D agents query models across retrieval, summarization, and visualization tasks.
3. **Cost-performance balance** – Use the cookbook's benchmarks to assign lightweight models for background indexing while reserving powerful models for user-facing interactions.

