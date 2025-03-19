from models.clip_model import CLIPModel
from unlearning.gradient_ascent import gradient_ascent_unlearning
from unlearning.random_labels import fine_tune_random_labels
from unlearning.kl_divergence import kl_divergence_unlearning
from unlearning.fine_tuning import fine_tune_clip
from unlearning.knowledge_distillation import knowledge_distillation_unlearning
from evaluation.metrics import cosine_similarity_change, retrieval_accuracy

# Load models
teacher_model = CLIPModel()
student_model = CLIPModel()

# Sample forget & retain sets
forget_images = ["cat.jpg", "dog.jpg"]
forget_texts = ["A cat sitting on a sofa", "A dog running in a park"]
retain_images = ["car.jpg", "tree.jpg"]
retain_texts = ["A red sports car", "A tree in autumn"]

# Apply unlearning methods
img_emb_before, txt_emb_before = teacher_model.get_image_embedding(forget_images[0]), teacher_model.get_text_embedding(forget_texts[0])
img_emb_after, txt_emb_after = gradient_ascent_unlearning(teacher_model, forget_images, forget_texts)

# Fine-Tuning Baseline
fine_tuned_model = fine_tune_clip(teacher_model, retain_images, retain_texts)

# Knowledge Distillation Baseline
distilled_model = knowledge_distillation_unlearning(teacher_model, student_model, retain_images, retain_texts)

# Evaluate
print("Cosine Similarity Drop (GAU):", cosine_similarity_change(img_emb_before, img_emb_after))
print("Retrieval Accuracy Change (Fine-Tuning):", retrieval_accuracy(img_emb_before, img_emb_after, retain_images))
print("Retrieval Accuracy Change (Distillation):", retrieval_accuracy(img_emb_before, img_emb_after, retain_images))
