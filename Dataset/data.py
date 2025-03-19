import os
import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image
import torch
import open_clip

#%% Define dataset paths
DATASET_DIR = "datasets"
FLICKR_DIR = os.path.join(DATASET_DIR, "flickr30k")
LAION_DIR = os.path.join(DATASET_DIR, "laion_subset")
os.makedirs(FLICKR_DIR, exist_ok=True)
os.makedirs(LAION_DIR, exist_ok=True)

#%% Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B/32")
model.to(device)

#%% Download Flickr30K dataset.
def download_flickr30k():
    flickr_url = "http://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip"
    zip_path = os.path.join(FLICKR_DIR, "flickr30k.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading Flickr30K...")
        response = requests.get(flickr_url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete!")

#%% Extract a 50K subset from LAION-400M.
def extract_laion_subset():
    laion_metadata_url = "https://huggingface.co/datasets/laion/laion400m-metadatas/raw/main/laion400m-meta.parquet"
    metadata_path = os.path.join(LAION_DIR, "laion400m-meta.parquet")

    if not os.path.exists(metadata_path):
        print("Downloading LAION-400M metadata...")
        response = requests.get(laion_metadata_url, stream=True)
        with open(metadata_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

    df = pd.read_parquet(metadata_path)
    subset = df.sample(n=50000, random_state=42)  # Take a 50K random subset
    subset.to_csv(os.path.join(LAION_DIR, "laion_subset.csv"), index=False)
    print("LAION-400M subset saved!")

#%% Process images and text into CLIP embeddings.
def preprocess_images_text(dataset_path, output_path):
    data = pd.read_csv(dataset_path)
    image_embeds = []
    text_embeds = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_url = row["url"]
        text = row["text"]
        
        try:
            # Download image
            response = requests.get(image_url, stream=True)
            img = Image.open(response.raw).convert("RGB")
            img = preprocess(img).unsqueeze(0).to(device)
            
            # Extract embeddings
            with torch.no_grad():
                image_embed = model.encode_image(img)
                text_embed = model.encode_text(tokenizer([text]).to(device))

            image_embeds.append(image_embed.cpu().numpy())
            text_embeds.append(text_embed.cpu().numpy())

        except Exception as e:
            print(f"Skipping image {image_url}: {e}")

    # Save embeddings
    torch.save({"images": image_embeds, "texts": text_embeds}, output_path)
    print(f"Processed embeddings saved to {output_path}")

# Run preprocessing
download_flickr30k()
extract_laion_subset()
preprocess_images_text(os.path.join(LAION_DIR, "laion_subset.csv"), os.path.join(LAION_DIR, "laion_embeddings.pth"))
#%%