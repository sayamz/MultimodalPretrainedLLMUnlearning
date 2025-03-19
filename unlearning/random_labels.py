import random

#%% Fine-tunes CLIP by replacing target image captions with random labels
def fine_tune_random_labels(model, forget_images, dataset_texts, lr=0.001, steps=10):
    model.train()
    
    random_texts = random.sample(dataset_texts, len(forget_images))  # Pick random captions
    forget_image_embeds = torch.cat([model.get_image_embedding(img) for img in forget_images])
    random_text_embeds = torch.cat([model.get_text_embedding(txt) for txt in random_texts])

    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.cosine_similarity(forget_image_embeds, random_text_embeds).mean()
        loss.backward()
        optimizer.step()

    return model

#%%