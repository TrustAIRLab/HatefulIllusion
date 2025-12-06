import os, sys, re, json
import numpy as np
import pandas as pd
import torch
from utils import *
import random
from datasets import load_dataset
from huggingface_hub import snapshot_download
import time

removed_messages = ["Chang", "Savage", "Moon_Man_0", "Moon_Man_1", "Anti_Antifa_1", "Misogyny_1", "Anti_LGBTQ_1", "Anti_LGBTQ_0"]

class OpticalIllusionDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 target="digits",
                 image_root="data/HatefulIllusion_Dataset",
                 visibility_label=None
                ):
        
        self.target = target
        self.image_root = image_root

        # check the dataset preparation
        if not os.path.exists(self.image_root):
            repo_id = "yiting/HatefulIllusion_Dataset"
            snapshot_download(repo_id, 
                            repo_type="dataset",
                            local_dir=self.image_root)
            
            print("HatefulIllusion dataset downloaded to:", image_root)
        else:
            pass
    
        self.metadata = load_dataset(self.image_root, target)["train"]
        self.image_root = os.path.join(image_root, target)
        
        data = []
        
        for item in self.metadata:
            message = str(item["message"])
            if any(removed_msg in message for removed_msg in removed_messages):
                continue
  
            data.append({
                "image_fname": os.path.join(self.image_root, item["image"]),
                "message": item["message"],
                "condition_image": os.path.join(self.image_root, item["condition_image"]),
                "prompt": item["prompt"],
                "visibility": int(item["visibility"]),
            })
        
        if len(visibility_label)>0:
            self.data = [item for item in data if item["visibility"] in visibility_label]
        else:
            self.data = data
    
    def __getitem__(self, idx):
        
        return self.data[idx] # image_fname, message, condition_image, prompt, visibility
    
    def __len__(self):
        return len(self.data)

class MessageImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 target="digits",
                 image_root="data/HatefulIllusion_Dataset",
                ):
        
        self.target = target
        self.image_root = os.path.join(image_root, target)
        
        image_fnames = os.listdir(os.path.join(self.image_root, "messages"))
        self.image_fnames = [img for img in image_fnames if not any(item in img for item in removed_messages)]
        
    def __getitem__(self, idx):
        return {"image_fname": os.path.join(self.image_root, "messages", self.image_fnames[idx]), "message": self.image_fnames[idx].replace(".png", "")}
    
    def __len__(self):
        return len(self.image_fnames)