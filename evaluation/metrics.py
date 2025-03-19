import torch

#%% Measures the drop in cosine similarity before & after unlearning
def cosine_similarity_change(before, after):
    return torch.cosine_similarity(before, after).mean().item()

#%% Computes retrieval accuracy before and after unlearning
def retrieval_accuracy(before_embeddings, after_embeddings, dataset):
    correct = 0
    for emb in before_embeddings:
        similarity = torch.cosine_similarity(emb.unsqueeze(0), after_embeddings)
        if similarity.argmax().item() == dataset.index(emb):
            correct += 1
    return correct / len(dataset)
