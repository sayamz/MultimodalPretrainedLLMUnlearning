import torch

#%%
import open_clip
from PIL import Image

print(open_clip.__version__)
#%%
class CLIPModel:
    def __init__(self, model_name="ViT-B/32"):
        """Initialize CLIP model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)

    def get_image_embedding(self, image_path):
        """Extract image embedding"""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(image)

    def get_text_embedding(self, text):
        """Extract text embedding"""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens)

#%%
