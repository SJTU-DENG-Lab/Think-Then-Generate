<div align="center">

# Think-Then-Generate: <br> Reasoning-Aware Text-to-Image Diffusion with LLM Encoders

[![arXiv](https://img.shields.io/badge/arXiv-2512.23576-b31b1b.svg)](xxx)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-ffd21e)](https://huggingface.co/SJTU-Deng-Lab/Think-Then-Generate-T2I/tree/main)
[![Project Page](https://img.shields.io/badge/Website-Gallery-blue)](https://zhijie-group.github.io/Think-Then-Generate/)

<img src="assets/pipeline.png" width="100%" alt="Think-Then-Generate Teaser">

</div>

---

## 🧠 How it Works

Traditional text-to-image models often map words to pixels without truly understanding the implicit semantics behind a prompt. To break this limitation, our pipeline operates in three key stages:

1.  **Chain-of-Thought (CoT) Reasoning** 
    * By fine-tuning **Qwen2.5-VL**, we activate the model's reasoning capabilities.
    * Instead of blindly generating, the model analyzes the user's request, breaking down scene composition, lighting, and object relationships to formulate a "refined prompt."

2.  **Dual-GRPO Reinforcement Learning** 
    * We introduce a collaborative RL strategy where the LLM encoder and the DiT (Diffusion Transformer) generator evolve together.
    * The LLM learns to draft better instructions, while the DiT enhances its rendering fidelity based on direct visual feedback.

3.  **Bridging Logic and Vision** 
    * The optimized prompt acts as a semantic bridge, ensuring the final output is a deep, accurate realization of the user's original intent—especially for complex logic like multi-step instructions or spatial reasoning.

## 🛠️ Installation

Install the necessary dependencies:

```bash
pip install torch transformers diffusers accelerate
```

## 🚀 Inference

Run the model on a single GPU to experience reasoning-aware generation.

```bash
python inference.py \
  --model_path "SJTU-Deng-Lab/Think-Then-Generate-T2I" \
  --prompt "A multi-panel illustration showing the story of marking the boat to find a sword, with clear steps from dropping the sword to carving a mark on the boat." \
  --output "sword_result.jpg"

```


## 🖊️ Citation

If you find our work helpful, please consider citing:

```bibtex
@article{xxx,
  title={Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders},
  author={Authors},
  journal={arXiv preprint},
  year={2025}
}

```
