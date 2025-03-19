import torch

#%%
def gradient_ascent_unlearning(model, forget_images, forget_texts, lr=0.01, steps=10):
    """Apply gradient ascent to unlearn specific text-image pairs"""
    model.train()
    
    image_embeds = torch.cat([model.get_image_embedding(img) for img in forget_images])
    text_embeds = torch.cat([model.get_text_embedding(txt) for txt in forget_texts])

    image_embeds.requires_grad = True
    text_embeds.requires_grad = True
    optimizer = torch.optim.SGD([image_embeds, text_embeds], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.cosine_similarity(image_embeds, text_embeds).mean()
        (-loss).backward()  # Reverse gradient direction
        optimizer.step()

    return image_embeds.detach(), text_embeds.detach()

#%%