import torch

#%% Fine-tunes CLIP on the retain set after removing the forget set
def fine_tune_clip(model, retain_images, retain_texts, lr=0.001, epochs=5):
    model.train()
    
    retain_image_embeds = torch.cat([model.get_image_embedding(img) for img in retain_images])
    retain_text_embeds = torch.cat([model.get_text_embedding(txt) for txt in retain_texts])

    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -torch.cosine_similarity(retain_image_embeds, retain_text_embeds).mean()
        loss.backward()
        optimizer.step()

    return model
