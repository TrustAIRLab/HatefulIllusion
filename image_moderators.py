import base64
from io import BytesIO
import requests
import json
import argparse
from pathlib import Path
import os
import pandas as pd
import torch
from PIL import Image
import requests
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
from torch.nn import functional as F
import open_clip
import pickle
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from huggingface_hub import snapshot_download

class Q16(torch.nn.Module):
    def __init__(self, checkpoint_dir="checkpoints/Q16/prompts.p", device="cuda"):
        super(Q16, self).__init__()
        
        self.device = device
        model_name, pretrained = "ViT-L-14", "openai"
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(torch.float32).to(self.device)

        # download checkpoints if not exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            print(f"Downloading Q16 checkpoints from Hugging Face Hub to {checkpoint_dir}...")
            snapshot_download(repo_id="yiting/Q16",
                    repo_type="model",
                    local_dir=os.path.dirname(checkpoint_dir))
        
        embeddings = torch.FloatTensor(pickle.load(open(checkpoint_dir, 'rb')))
        self.prompts = torch.nn.Parameter(embeddings).to(self.device)
    
    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(img_path)) for img_path in image_paths]
        images = torch.stack(images)
        return images
    
    @ torch.no_grad()
    def classify(self, x):
        x = x.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feat @ text_feat.T)
        return similarity.squeeze()
    
    @ torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feat @ text_feat.T)
        
        logits = F.softmax(similarity, dim=-1) # add this to adapt adversarial attack
        return logits
    
    @ torch.no_grad()
    def forward_head_only(self, e): # input is image features: [bs, embed_len]
        e /= e.norm(dim=-1, keepdim=True)
        e = e.to(self.device)
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        similarity = (100.0 * e @ text_feat.T)
        logits = F.softmax(similarity, dim=-1)
        return logits

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = torch.nn.functional.normalize(image_embeds)
    normalized_text_embeds = torch.nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = torch.nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = torch.nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = torch.nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = torch.nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    def preprocess_images(self, image_paths):
        images = [Image.open(img_path) for img_path in image_paths]
        safety_checker_input = self.safety_feature_extractor(images, return_tensors="pt") # pixel_values: [1, 3, 224, 224]
        images = safety_checker_input.pixel_values
        return images
    
    @torch.no_grad()
    def classify(self, clip_input):
        clip_input = clip_input.to(self.vision_model.device)
        
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]
        
        # use logits
        one_hot_list = [[0, 1] if x else [1, 0] for x in has_nsfw_concepts]
        logits = torch.tensor(one_hot_list)
        return logits

def encode_image(image_path):
    # Open the image
    with Image.open(image_path) as img:
        # Convert to RGB if the image is not already in RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Save the image to a bytes buffer
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        # Encode the image in base64
        base64_bytes = base64.b64encode(buffered.getvalue())
        base64_str = base64_bytes.decode("utf-8")
    return base64_str


def load_image_moderators(model_name, **gen_kwargs):
    
    if "omni" in model_name:
        from openai import OpenAI
        client = OpenAI(api_key=gen_kwargs["api_key"])
        return client
    
    # pip install google-cloud
    # pip install google-cloud-vision
    elif "SafeSearch" in model_name:
        from google.oauth2 import service_account
        from google.cloud import vision
        
        config_file = ".env/google_cloud.json"
        if not os.path.exists(config_file):
            raise FileNotFoundError("Please download the credential file from Google Cloud first.")

        creds = service_account.Credentials.from_service_account_file(config_file)
        client = vision.ImageAnnotatorClient(credentials=creds)

        return client
    
    elif "azure" == model_name:
        from azure.ai.contentsafety import ContentSafetyClient
        from azure.ai.contentsafety.models import ImageCategory
        from azure.core.credentials import AzureKeyCredential
        from azure.core.exceptions import HttpResponseError
        from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData

        key = gen_kwargs["api_key"]
        endpoint = gen_kwargs["endpoint"]

        if key is None or endpoint is None:
            raise FileNotFoundError("Please setup your key and endpoint for axure content safety detector.")
        
        # Create a Content Safety client
        client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
        return client
    
    elif "azure_multimodal" == model_name:
        return ""

    elif model_name == "q16":
        model = Q16(device="cuda")
        model.clip_model.eval()
        model.clip_model.requires_grad_(False)
        return model
    
    elif model_name == "safety_checker":
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        model = StableDiffusionSafetyChecker.from_pretrained(safety_model_id).to("cuda")
        
        model.vision_model.eval()
        model.visual_projection.eval()
        model.vision_model.requires_grad_(False)
        model.visual_projection.requires_grad_(False)
        
        return model
    
def moderate(model_name, pipeline, image_path, **gen_kwargs):
    
    if "omni" in model_name:
        base64_image = encode_image(image_path)
                
        response = pipeline.moderations.create(
            model="omni-moderation-latest",
            input=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                },
            ],
        )
        output = response.results[0]
        output = output.flagged
        
        return output
    
    elif "SafeSearch" in model_name:

        from google.cloud import vision

        with open(image_path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = pipeline.safe_search_detection(image=image)
        response = response.safe_search_annotation
        likelihood_name = (
            "UNKNOWN",
            "VERY_UNLIKELY",
            "UNLIKELY",
            "POSSIBLE",
            "LIKELY",
            "VERY_LIKELY",
            )
        class_names = ["adult", "medical", "spoof", "violence", "racy"]
        response = {key: likelihood_name[getattr(response, key)] for key in class_names}
        
        return response
    
    elif "azure" == model_name:
        from azure.ai.contentsafety import ContentSafetyClient
        from azure.ai.contentsafety.models import ImageCategory
        from azure.core.credentials import AzureKeyCredential
        from azure.core.exceptions import HttpResponseError
        from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
        
        with open(image_path, "rb") as file:
            request = AnalyzeImageOptions(image=ImageData(content=file.read()))

        try:
            response = pipeline.analyze_image(request)
        except HttpResponseError as e:
            print("Analyze image failed.")
            if e.error:
                print(f"Error code: {e.error.code}")
                print(f"Error message: {e.error.message}")
                raise
            print(e)
            raise

        hate_result = next(item for item in response.categories_analysis if item.category == ImageCategory.HATE)
        self_harm_result = next(item for item in response.categories_analysis if item.category == ImageCategory.SELF_HARM)
        sexual_result = next(item for item in response.categories_analysis if item.category == ImageCategory.SEXUAL)
        violence_result = next(item for item in response.categories_analysis if item.category == ImageCategory.VIOLENCE)

        response = {"Hate": hate_result.severity, 
                    "SelfHarm": self_harm_result.severity,
                    "Sexual": sexual_result.severity,
                    "Violence": violence_result.severity}
        
        return response
    
    elif "azure_multimodal" == model_name:
        import requests
        
        # Replace with your actual endpoint and subscription key
        key = gen_kwargs["api_key"]
        endpoint = gen_kwargs["endpoint"]
        url = f"{endpoint}/contentsafety/imageWithText:analyze?api-version=2024-09-15-preview"
        
        if key is None or endpoint is None:
            raise FileNotFoundError("Please setup your key and endpoint for axure content safety detector.")
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/json"
        }

        payload = {
            "image": {
                "content": encoded_string
            },
            "categories": ["Hate"],
            "enableOcr": True,
            "text": None
        }

        response = requests.post(url, headers=headers, json=payload)
        response = response.json()
        return response

    elif "q16" in model_name or "safety_checker" in model_name:
        images = pipeline.preprocess_images([image_path])
        with torch.no_grad():
            logits = pipeline.classify(images)
        preds = torch.argmax(logits).detach().cpu().numpy().tolist()
        return preds
        

