import torch
import open_clip
from utils.data import LAIONDataset, custom_collate
#%%
from torch.utils.data import DataLoader
from utils.visualization import plot_metrics
from unlearning.methods import gradient_ascent_unlearning, fine_tuning_random_labels, kl_divergence_minimization

#%%
from evaluation.metrics import (
    evaluate_perplexity_shift,
    evaluate_cosine_similarity_drop,
    evaluate_retrieval_accuracy_change,
    evaluate_generalization_retention,
    evaluate_forget_set_performance
)

#%% ========== Unlearning and Evaluation ==========
def run_unlearning_pipeline(model, forget_set, retain_set, method_fn, label):
    print(f"Unlearning model with {label}...")
    model = method_fn(model, forget_set, retain_set)
    print(f"Evaluating {label}...")

    metrics = {
        "Perplexity Shift": evaluate_perplexity_shift(model, retain_loader),
        "Cosine Similarity Drop": evaluate_cosine_similarity_drop(model, retain_loader),
        "Retrieval Accuracy Change": evaluate_retrieval_accuracy_change(model, retain_loader),
        "Generalization Retention": evaluate_generalization_retention(model, retain_loader),
        "Forget Set Performance": evaluate_forget_set_performance(model, forget_loader)
    }

    print("\n".join([f"{k}: {v}" for k, v in metrics.items()]))
    return model, metrics
#%%
# Set device and model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'laion400m_e32'
#PRETRAINED='laion2b_e16'

#%%
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model.to(DEVICE)

#open_clip.list_pretrained()
#%%
# Paths
CSV_PATH = 'Dataset/laion_subset/Laion_subset.csv'

# Load Datasets and DataLoaders
forget_loader = DataLoader(LAIONDataset(CSV_PATH, preprocess_train, forget_set=True), batch_size=32, shuffle=True, collate_fn=custom_collate)
retain_loader = DataLoader(LAIONDataset(CSV_PATH, preprocess_train, retain_set=True), batch_size=32, shuffle=True, collate_fn=custom_collate)


#%%
# Run all methods
model_ga, metrics_ga = run_unlearning_pipeline(model, forget_loader, retain_loader, gradient_ascent_unlearning, "Gradient Ascent Unlearning")
model_frl, metrics_frl = run_unlearning_pipeline(model, forget_loader, retain_loader, fine_tuning_random_labels, "Fine-Tuning Random Labels")
model_kld, metrics_kld = run_unlearning_pipeline(model, forget_loader, retain_loader, kl_divergence_minimization, "GA + KL Divergence")
#%%

plot_metrics({
    "Gradient Ascent Unlearning": metrics_ga,
    "Fine-Tuning Random Labels": metrics_frl,
    "GA + KL Divergence": metrics_kld
})
#%%