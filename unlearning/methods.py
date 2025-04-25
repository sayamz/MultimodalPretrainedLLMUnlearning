import torch
from torch import nn
import open_clip
from transformers import CLIPProcessor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Define unlearning techniques
def gradient_ascent_unlearning(model, forget_set, retain_set=None, epochs=10, lr=0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for images, texts in forget_set:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            texts = open_clip.tokenize(texts).to(device)
            optimizer.zero_grad()
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            loss = -torch.tensor(cosine_similarity(image_features.cpu().detach().numpy(), text_features.cpu().detach().numpy()).mean(), device=device, requires_grad=True)
            loss.backward()
            optimizer.step()
    return model

def fine_tuning_random_labels(model, forget_set, retain_set=None, epochs=10, lr=0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for images, texts in forget_set:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            random_texts = open_clip.tokenize(np.random.choice(texts, len(texts))).to(device)
            optimizer.zero_grad()
            image_features = model.encode_image(images)
            text_features = model.encode_text(random_texts)
            loss = torch.tensor(cosine_similarity(image_features.cpu().detach().numpy(), text_features.cpu().detach().numpy()).mean(), device=device, requires_grad=True)
            loss.backward()
            optimizer.step()
    return model

def kl_divergence_minimization(model, forget_set, retain_set=None, epochs=10, lr=0.01):    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        # gradient ascent on the forget set
        for images, texts in forget_set:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            texts = open_clip.tokenize(texts).to(device)
            optimizer.zero_grad()
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            loss = -torch.tensor(cosine_similarity(image_features.cpu().detach().numpy(), text_features.cpu().detach().numpy()).mean(), device=device, requires_grad=True)
            loss.backward()
            optimizer.step()
            
        #KL-divergence minimization on the retain set
        for images, texts in retain_set:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            texts = open_clip.tokenize(texts).to(device)
            optimizer.zero_grad()
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            kl_loss = torch.nn.functional.kl_div(image_features.log_softmax(dim=-1), text_features.softmax(dim=-1), reduction='batchmean')
            kl_loss.backward()
            optimizer.step()
    return model
