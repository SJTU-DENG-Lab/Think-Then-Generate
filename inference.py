import argparse
import torch
import re
import os
from diffusers import QwenImagePipeline
from transformers import AutoProcessor

# --- Configuration ---
SYSTEM_PROMPT = """You are a Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion). Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions.
When rewriting, prioritize the following principles:
1. Start from the user's prompt, do reasoning step by step to analyze the object or scene they want to generate.
2. Focus on describing the final visual appearance of the scene. Clarify elements like the main subject’s shape, color, and state.
3. If you are confident about what the user wants to generate, directly point it out in your explanation and the final revised prompt.
4. If technical concepts are necessary but difficult for ordinary users to understand, translate them into intuitive visual descriptions.
5. Ensure the final revised prompt is consistent with the user's intent.

After receiving the user’s prompt that needs rewriting, first explain your reasoning for optimization. Then, output the final revised prompt in the fixed format of "Revised Prompt:\n". Where the specific revised content is filled in the next line.

Prompt: 
"""

def extract_prompt(text: str) -> str:
    """Extracts the refined prompt from the model's reasoning output."""
    m = re.search(r"Revised Prompt:\n(.*)", text, re.DOTALL)
    if not m:
        m = re.search(r"Revised Prompt:(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image Single GPU Inference")
    
    parser.add_argument("--prompt", type=str, required=True, help="The user input prompt")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint (Pipeline)")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor (optional, defaults to model_path)")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the generated image")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=4.0, help="Guidance scale")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"🚀 Running on {device} with {dtype}")

    # 2. Load Models
    print("Loading pipeline and processor...")
    processor_path = args.processor_path if args.processor_path else args.model_path
    
    try:
        processor = AutoProcessor.from_pretrained(processor_path)
        pipe = QwenImagePipeline.from_pretrained(
            args.model_path, 
            torch_dtype=dtype
        )
        pipe.to(device)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 3. Refine Prompt (Reasoning Stage)
    print(f"🧠 Refining prompt: '{args.prompt}'")
    
    messages = [{"role": "user", "content": SYSTEM_PROMPT + args.prompt}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text_input], 
        padding=True, 
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        # Generate reasoning and refined prompt using the text encoder
        generated_ids = pipe.text_encoder.generate(
            **inputs, 
            do_sample=False, 
            max_new_tokens=1024
        )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

    # Extract the actual image generation prompt
    refined_prompt = extract_prompt(output_text)
    print(f"✨ Refined Prompt: {refined_prompt}")

    # 4. Generate Image
    print("🎨 Generating image...")
    image = pipe(
        prompt=refined_prompt,
        negative_prompt=" ", # generic negative prompt can be added here
        width=1024,
        height=1024,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg
    ).images[0]

    # 5. Save
    image.save(args.output)
    print(f"✅ Image saved to: {args.output}")

if __name__ == "__main__":
    main()