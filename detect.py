import base64
import requests
import json
import argparse
from pathlib import Path
import os, sys
import time
import pandas as pd
from my_datasets import OpticalIllusionDataset, MessageImageDataset
from vlms import *
from image_moderators import *
import re
from utils import *

vlm_names = ["gpt-4v", "gpt-4o", "gemini", "gemini_2", "cogvlm", "cogvlm2", "llava", "llava_v0", "qwen", "qwen_2b"]
image_moderators = ["q16", "safety_checker", "omni", "SafeSearch", "azure", "azure_multimodal"]

def query(args):
    
    args.save_path = os.path.join(args.save_path, args.model_name)
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.query_mode == "zero-shot":
        detect_prompts = open("data/prompts/detection_zero-shot.txt", "r").read().splitlines()
    elif args.query_mode == "cot":
        detect_prompts = open("data/prompts/detection_cot.txt", "r").read()
        detect_prompts = [detect_prompts]
    else:
        raise ValueError("wrong query mode, use zero-shot or cot")
    
    # load vlm
    gen_kwargs = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens
    }
    
    if "gpt" in args.model_name or "gemini" in args.model_name or "omni" in args.model_name:
        gen_kwargs["api_key"] = args.api_key
    
    if "azure" in args.model_name:
        gen_kwargs["api_key"] = args.api_key
        gen_kwargs["endpoint"] = args.endpoint
    
    if args.model_name in vlm_names:
        pipeline = load_vlm(model_name=args.model_name, **gen_kwargs)
    elif args.model_name in image_moderators:
        pipeline = load_image_moderators(model_name=args.model_name, **gen_kwargs)
    else:
        raise Exception(f"It doesn't support {args.model_name}")
        
    # if the result has been saved
    if os.path.exists(os.path.join(args.save_path, f"{args.target}_{args.dataset_name}.json")):
        result = json.load(open(os.path.join(args.save_path, f"{args.target}_{args.dataset_name}.json"), "r"))
    else:
        result = []
    
    saved_indices = [item["idx"] for item in result]
    
    # load data
    if args.dataset_name == "illusion":
        dataset = OpticalIllusionDataset(target=args.target,
                                        visibility_label=[]) # all visibility labels
    elif args.dataset_name == "message":
        dataset = MessageImageDataset(target=args.target)
    
    idx = -1

    for prompt_idx, detect_prompt in enumerate(detect_prompts):
        print("detection prompts:", detect_prompt)
        
        for item in dataset:
            
            idx += 1
            image_fname = item["image_fname"]
            message = item["message"]
            
            if str(idx) in saved_indices:
                print("data already saved!")
                continue
            
            if args.model_name in vlm_names:
                response = inference(args.model_name, pipeline, image_fname, detect_prompt, **gen_kwargs)
            elif args.model_name in image_moderators:
                response = moderate(args.model_name, pipeline, image_fname, **gen_kwargs)
            else:
                raise Exception(f"It doesn't support {args.model_name}")
            
            print(message, response)
        
            result.append({
                "idx": str(idx),
                "image_fname": image_fname,
                "prompt": detect_prompt,
                "response": response
            })

            json.dump(result, open(os.path.join(args.save_path, f"{args.target}_{args.dataset_name}.json"), "w"), indent=2)
            
            if "gemini" in args.model_name:
                time.sleep(6)
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # customized
    parser.add_argument("--dataset_name", type=str, default="illusion", choices=["illusion", "message"])
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--query_mode", type=str, default="zero-shot", choices=["zero-shot", "cot"])
    parser.add_argument("--target", type=str, default="hate_symbols")
    parser.add_argument("--save_path", type=str, default="outputs/classification")
    parser.add_argument("--image_size", type=int, default=1024)
    
    # setup
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    
    # temperature
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    query(args)