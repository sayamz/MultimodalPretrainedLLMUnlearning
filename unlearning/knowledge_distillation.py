import torch.nn.functional as F

#%% Forces knowledge distillation between teacher and student model to unlearn specific data
def knowledge_distillation_unlearning(teacher_model, student_model, retain_images, retain_texts, temperature=1.0, alpha=0.7):
    student_model.train()
    
    retain_image_embeds = torch.cat([student_model.get_image_embedding(img) for img in retain_images])
    retain_text_embeds = torch.cat([student_model.get_text_embedding(txt) for txt in retain_texts])

    with torch.no_grad():
        teacher_image_embeds = torch.cat([teacher_model.get_image_embedding(img) for img in retain_images])
        teacher_text_embeds = torch.cat([teacher_model.get_text_embedding(txt) for txt in retain_texts])

    # Compute distillation loss
    loss = alpha * F.kl_div(retain_image_embeds.log_softmax(dim=-1) / temperature, 
                             teacher_image_embeds.softmax(dim=-1), reduction='batchmean') + \
           (1 - alpha) * F.kl_div(retain_text_embeds.log_softmax(dim=-1) / temperature, 
                                  teacher_text_embeds.softmax(dim=-1), reduction='batchmean')

    optimizer = torch.optim.Adam(student_model.model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return student_model

#%%