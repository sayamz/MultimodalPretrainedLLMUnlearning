import torch.nn.functional as F

#%% Forces KL-Divergence between original and updated model to unlearn specific data
def kl_divergence_unlearning(model, forget_images, forget_texts, baseline_model, alpha=0.5):
    model.train()
    
    forget_image_embeds = torch.cat([model.get_image_embedding(img) for img in forget_images])
    forget_text_embeds = torch.cat([model.get_text_embedding(txt) for txt in forget_texts])
    
    with torch.no_grad():
        baseline_img_embeds = torch.cat([baseline_model.get_image_embedding(img) for img in forget_images])
        baseline_text_embeds = torch.cat([baseline_model.get_text_embedding(txt) for txt in forget_texts])

    loss = alpha * F.kl_div(forget_image_embeds.log_softmax(dim=-1), baseline_img_embeds.softmax(dim=-1), reduction='batchmean') + \
           (1 - alpha) * F.kl_div(forget_text_embeds.log_softmax(dim=-1), baseline_text_embeds.softmax(dim=-1), reduction='batchmean')

    optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model
