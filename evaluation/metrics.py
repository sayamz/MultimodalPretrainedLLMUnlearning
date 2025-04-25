import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np


# Define evaluation metrics
def evaluate_perplexity_shift(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_loss = 0
    for images, texts in data_loader:
        if images.size(0) == 0:
            continue
        images = images.to(device)
        texts = open_clip.tokenize(texts).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            loss = torch.nn.functional.cross_entropy(image_features, text_features)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_cosine_similarity_drop(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_similarity = 0
    for images, texts in data_loader:
        if images.size(0) == 0:
            continue
        images = images.to(device)
        texts = open_clip.tokenize(texts).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            similarity = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy()).mean()
            total_similarity += similarity
    return total_similarity / len(data_loader)

def evaluate_retrieval_accuracy_change(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    correct = 0
    total = 0
    for images, texts in data_loader:
        if images.size(0) == 0:
            continue
        images = images.to(device)
        texts = open_clip.tokenize(texts).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            similarities = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy())
            predictions = np.argmax(similarities, axis=1)
            correct += (predictions == np.arange(len(texts))).sum()
            total += len(texts)
    return correct / total

def evaluate_generalization_retention(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_loss = 0
    for images, texts in data_loader:
        if images.size(0) == 0:
            continue
        images = images.to(device)
        texts = open_clip.tokenize(texts).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            loss = torch.nn.functional.mse_loss(image_features, text_features)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_forget_set_performance(model, forget_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_similarity_drop_forget_set = 0
    for images, texts in forget_loader:
        if images.size(0) == 0:
            continue
        images = images.to(device)
        texts = open_clip.tokenize(texts).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            similarity = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy()).mean()
            total_similarity_drop_forget_set += similarity
    return total_similarity_drop_forget_set / len(forget_loader)

def evaluate_mia_resistance(model, data_loader):
    # Placeholder for MIA resistance evaluation
    # Implement your specific method for testing resistance to membership inference attacks
    pass