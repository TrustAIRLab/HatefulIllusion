import base64
import requests
import json
import argparse
from pathlib import Path
import os
import sys
import time
import pandas as pd
import torch
from PIL import Image
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import random

VLM2ModelPaths = {
    "gpt-4v": "gpt-4-turbo-2024-04-09",
    "gpt-4o": "gpt-4o",
    "gemini": "gemini-1.5-flash",
    "gemini_2": "gemini-2.0-flash-exp",
    # For local or HF models, replace these placeholders with your own paths or model ids
    "llava": "llava-hf/llava-v1.6-mistral-7b-hf", 
    "llava_v0": "llava-hf/llava-1.5-7b-hf",
    "llava_13b": "liuhaotian/llava-v1.6-vicuna-13b",
    "cogvlm": "zai-org/cogvlm-chat-hf",
    "cogvlm2": "zai-org/cogvlm2-llama3-chat-19B",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen_2b": "Qwen/Qwen2-VL-2B-Instruct",
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    import google.generativeai as genai

    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def get_image_type(image_path):
    """Returns the type of the image (for example, JPEG, PNG)."""
    with Image.open(image_path) as img:
        return img.format.lower()


def load_vlm(model_name, **gen_kwargs):
    if "gpt" in model_name:
        from openai import OpenAI

        # Expect the caller to pass api_key in gen_kwargs or use OPENAI_API_KEY env var
        api_key = gen_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI api key not provided. Set OPENAI_API_KEY or pass api_key in gen_kwargs.")
        client = OpenAI(api_key=api_key)
        return client

    elif "gemini" in model_name:
        import google.generativeai as genai

        api_key = gen_kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("Gemini api key not provided. Set GEMINI_API_KEY or pass api_key in gen_kwargs.")

        new_gen_kwargs = {
            "temperature": gen_kwargs.get("temperature", 1.0),
            "max_output_tokens": gen_kwargs.get("max_new_tokens", 1024),
        }

        genai.configure(api_key=api_key)
        checkpoint = VLM2ModelPaths[model_name]
        model = genai.GenerativeModel(
            model_name=checkpoint,
            generation_config=new_gen_kwargs,
        )
        return model

    elif model_name == "llava":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        model = LlavaNextForConditionalGeneration.from_pretrained(
            VLM2ModelPaths[model_name],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        return model

    elif model_name == "llava_v0":
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            VLM2ModelPaths[model_name],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        return model

    elif "cogvlm" in model_name:
        from transformers import AutoModelForCausalLM, LlamaTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            VLM2ModelPaths[model_name],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda").eval()
        return model

    elif "qwen" in model_name:
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM2ModelPaths[model_name],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        return model


def inference(model_name, pipeline, image_path, prompt, **gen_kwargs):
    if "gpt" in model_name:
        gen_kwargs_openai = {
            "temperature": gen_kwargs.get("temperature", 1.0),
            "max_tokens": gen_kwargs.get("max_new_tokens", 1024),
        }

        base64_image = encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            }
        ]
        completion = pipeline.chat.completions.create(
            model=VLM2ModelPaths[model_name],
            store=True,
            messages=messages,
            **gen_kwargs_openai,
        )
        output = completion.choices[0].message.content
        return output

    elif "gemini" in model_name:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        image = Image.open(image_path)
        output = pipeline.generate_content(
            [image, prompt],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        return output.text

    elif model_name in ["llava", "llava_13b"]:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        processor = LlavaNextProcessor.from_pretrained(VLM2ModelPaths[model_name])
        processor.tokenizer.padding_side = "left"

        images = [Image.open(image_path)]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        conv_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images, conv_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = pipeline.generate(**inputs, **gen_kwargs)
        response = processor.decode(output[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1]

    elif model_name == "llava_v0":
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(VLM2ModelPaths[model_name])

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        conv_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        images = [Image.open(image_path)]
        inputs = processor(images=images, text=conv_prompt, return_tensors="pt").to(0, torch.float16)
        with torch.no_grad():
            output = pipeline.generate(**inputs, **gen_kwargs)
        response = processor.decode(output[0], skip_special_tokens=True)
        return response.split("ASSISTANT:")[-1].strip()

    elif model_name == "cogvlm":
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

        if model_name == "cogvlm2":
            gen_kwargs["pad_token_id"] = 128002
            tokenizer = AutoTokenizer.from_pretrained(VLM2ModelPaths[model_name], trust_remote_code=True)
        else:
            tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

        target_image = Image.open(image_path).convert("RGB")

        inputs_conv = pipeline.build_conversation_input_ids(
            tokenizer, query=prompt, history=[], images=[target_image]
        )
        inputs = {
            "input_ids": inputs_conv["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": inputs_conv["token_type_ids"].unsqueeze(0).to("cuda"),
            "attention_mask": inputs_conv["attention_mask"].unsqueeze(0).to("cuda"),
            "images": [[inputs_conv["images"][0].to("cuda").to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = pipeline.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(outputs[0])
        return response

    elif model_name == "cogvlm2":
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

        gen_kwargs["pad_token_id"] = 128002
        tokenizer = AutoTokenizer.from_pretrained(VLM2ModelPaths[model_name], trust_remote_code=True)

        target_image = Image.open(image_path).convert("RGB")
        input_by_model = pipeline.build_conversation_input_ids(
            tokenizer,
            query=prompt,
            history=[],
            images=[target_image],
            template_version="chat",
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to("cuda"),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to("cuda"),
            "images": [[input_by_model["images"][0].to("cuda").to(torch.bfloat16)]]
            if target_image is not None
            else None,
        }
        with torch.no_grad():
            outputs = pipeline.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
        return response

    elif "qwen" in model_name:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(VLM2ModelPaths[model_name])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        with torch.no_grad() and torch.cuda.amp.autocast():
            generated_ids = pipeline.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


def icl_inference(model_name, pipeline, image_path, prompt, few_shot_examples, **gen_kwargs):
    if "gpt" in model_name:
        gen_kwargs_openai = {
            "temperature": 1.0,
            "max_tokens": 1024,
        }

        history = []
        if len(few_shot_examples) > 0:
            for idx, example in enumerate(few_shot_examples):
                demo_image_path = example["image_fname"]
                demo_response = example["response"]
                demo_image = encode_image(demo_image_path)

                if idx == 0:
                    history.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{demo_image}"}},
                            ],
                        }
                    )
                else:
                    history.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{demo_image}"}},
                            ],
                        }
                    )
                history.append({"role": "system", "content": demo_response})

            target_image = encode_image(image_path)
            history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_image}"}},
                    ],
                }
            )
        else:
            target_image = encode_image(image_path)
            history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_image}"}},
                    ],
                }
            )

        completion = pipeline.chat.completions.create(
            model=VLM2ModelPaths[model_name],
            store=True,
            messages=history,
            **gen_kwargs_openai,
        )
        output = completion.choices[0].message.content
        return output

    elif "gemini" in model_name:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        history = []
        if len(few_shot_examples) > 0:
            for idx, example in enumerate(few_shot_examples):
                demo_image_path = example["image_fname"]
                demo_response = example["response"]
                demo_image = Image.open(demo_image_path)

                if idx == 0:
                    history.append({"role": "user", "parts": [prompt, demo_image]})
                else:
                    history.append({"role": "user", "parts": [demo_image]})

                history.append({"role": "model", "parts": [demo_response]})

            target_image = Image.open(image_path)
            chat_session = pipeline.start_chat(history=history)
            output = chat_session.send_message(
                [target_image],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )

        else:
            target_image = Image.open(image_path)
            chat_session = pipeline.start_chat(history=history)
            output = chat_session.send_message(
                [prompt, target_image],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
        return output.text

    elif model_name == "llava":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        processor = LlavaNextProcessor.from_pretrained(VLM2ModelPaths[model_name])

        history = []
        images = []

        if len(few_shot_examples) > 0:
            for idx, example in enumerate(few_shot_examples):
                demo_image_path = example["image_fname"]
                demo_response = example["response"]
                demo_image = Image.open(demo_image_path)

                if idx == 0:
                    history.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image"},
                            ],
                        }
                    )
                else:
                    history.append({"role": "user", "content": [{"type": "image"}]})

                history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": demo_response},
                        ],
                    }
                )
                images.append(demo_image)

            target_image = Image.open(image_path)
            images.append(target_image)
            conversation = history
            conversation.append({"role": "user", "content": [{"type": "image"}]})
        else:
            target_image = Image.open(image_path)
            images.append(target_image)
            conversation = history
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            )

        conv_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        print(conv_prompt, len(images), history)
        inputs = processor(images=images, text=conv_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = pipeline.generate(**inputs, **gen_kwargs)
        response = processor.decode(output[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1]

    elif model_name == "llava_v0":
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(VLM2ModelPaths[model_name])

        history = []
        images = []

        if len(few_shot_examples) > 0:
            for idx, example in enumerate(few_shot_examples):
                demo_image_path = example["image_fname"]
                demo_response = example["response"]
                demo_image = Image.open(demo_image_path)

                if idx == 0:
                    history.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image"},
                            ],
                        }
                    )
                else:
                    history.append({"role": "user", "content": [{"type": "image"}]})

                history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": demo_response},
                        ],
                    }
                )
                images.append(demo_image)

            target_image = Image.open(image_path)
            images.append(target_image)
            conversation = history
            conversation.append({"role": "user", "content": [{"type": "image"}]})
        else:
            target_image = Image.open(image_path)
            images.append(target_image)
            conversation = history
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            )

        conv_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=images, text=conv_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = pipeline.generate(**inputs, **gen_kwargs)
        response = processor.decode(output[0], skip_special_tokens=True)
        return response.split("ASSISTANT:")[-1].strip()

    elif "cogvlm" in model_name:
        raise Exception("CogVLM series does not support icl few shot yet!")

    elif "qwen" in model_name:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(VLM2ModelPaths[model_name])

        history = []
        if len(few_shot_examples) > 0:
            for idx, example in enumerate(few_shot_examples):
                demo_image_path = example["image_fname"]
                demo_response = example["response"]

                if idx == 0:
                    history.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": demo_image_path},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    )
                else:
                    history.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": demo_image_path},
                            ],
                        }
                    )
                history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": demo_response},
                        ],
                    }
                )

            history.append({"role": "user", "content": [{"type": "image", "image": image_path}]})
        else:
            history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image_path},
                    ],
                }
            )

        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = processor(
            text=text,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = pipeline.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
