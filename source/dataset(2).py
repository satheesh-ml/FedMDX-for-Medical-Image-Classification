# ============================================================
# FEDERATED LEARNING PIPELINE FOR CHEST X-RAY CLASSIFICATION
# CNN-ViT | DP-SGD | GSF Aggregation | Grad-CAM++ | SHAP
# ============================================================

import os
import cv2
import copy
import shap
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import zscore

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "MAIN_FOLDER"      : r"train",        # ← Change this
    "IMG_SIZE"         : 224,
    "BATCH_SIZE"       : 8,
    "LOCAL_EPOCHS"     : 2,
    "GLOBAL_ROUNDS"    : 3,
    "NUM_HOSPITALS"    : 3,
    "LEARNING_RATE"    : 1e-4,
    "DP_NOISE_MULT"    : 1.0,
    "DP_MAX_GRAD_NORM" : 1.0,
    "GSF_THRESHOLD"    : 2.0,
    "DEVICE"           : "cuda" if torch.cuda.is_available() else "cpu",
    "SAVE_DIR"         : "outputs",
}

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
DEVICE = CONFIG["DEVICE"]
print(f"Using device: {DEVICE}")

# ============================================================
# STEP 1 — LOAD IMAGES FROM PATIENT FOLDERS
# ============================================================

DISEASE_KEYWORDS = {
    "pneumonia"    : 0,
    "covid"        : 1,
    "tuberculosis" : 2,
    "normal"       : 3,
    "uncertain"    : 3,
}
CLASS_NAMES = ["Pneumonia", "COVID-19", "Tuberculosis", "Normal"]

def infer_label(path: str) -> int:
    text = path.lower().replace("\\", "/")
    for keyword, label in DISEASE_KEYWORDS.items():
        if keyword in text:
            return label
    return 3   # default → Normal

image_paths, raw_labels = [], []

# ── Collect all patient folders ──────────────────────────────
if not os.path.isdir(CONFIG["MAIN_FOLDER"]):
    raise FileNotFoundError(f"MAIN_FOLDER not found: {CONFIG['MAIN_FOLDER']}")

patient_folders = sorted([
    os.path.join(CONFIG["MAIN_FOLDER"], f)
    for f in os.listdir(CONFIG["MAIN_FOLDER"])
    if os.path.isdir(os.path.join(CONFIG["MAIN_FOLDER"], f))
])

if not patient_folders:
    raise FileNotFoundError("No sub-folders found inside MAIN_FOLDER.")

print(f"Patient Folders Found: {len(patient_folders)}")

for pf in patient_folders:
    for root, _, files in os.walk(pf):
        for file in sorted(files):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                fp = os.path.join(root, file)
                img_check = cv2.imread(fp)
                if img_check is not None:          # skip unreadable files
                    image_paths.append(fp)
                    raw_labels.append(infer_label(fp))
                else:
                    print(f"  ⚠ Skipped (unreadable): {fp}")

if len(image_paths) == 0:
    raise ValueError("No valid images found. Check MAIN_FOLDER path and image files.")

print(f"Total Valid Images   : {len(image_paths)}")

# ── Label distribution ───────────────────────────────────────
raw_labels = np.array(raw_labels)
unique_l, counts_l = np.unique(raw_labels, return_counts=True)
print("Label distribution:")
for u, c in zip(unique_l, counts_l):
    print(f"  {CLASS_NAMES[u]}: {c}")

# ── Drop classes with < 2 samples (can't stratify) ──────────
valid_classes = [u for u, c in zip(unique_l, counts_l) if c >= 2]
mask          = np.isin(raw_labels, valid_classes)
image_paths   = [image_paths[i] for i in range(len(image_paths)) if mask[i]]
raw_labels    = raw_labels[mask]

# Remap labels to 0-based contiguous
label_map   = {old: new for new, old in enumerate(sorted(set(raw_labels)))}
raw_labels  = np.array([label_map[l] for l in raw_labels])
CLASS_NAMES = [CLASS_NAMES[k] for k in sorted(label_map.keys())]
print(f"\nAfter filtering — Images: {len(image_paths)}  Classes: {CLASS_NAMES}")

# ============================================================
# STEP 2 — PREPROCESS  (Grayscale → 3-ch, Resize, Normalize)
# ============================================================

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

class XRayDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = list(paths)
        self.labels    = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            img = np.zeros((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3),
                           dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])

full_dataset = XRayDataset(image_paths, raw_labels, transform=preprocess)
print(f"Dataset ready: {len(full_dataset)} samples, {len(CLASS_NAMES)} classes")

# ============================================================
# STEP 3 — FEDERATED CLIENT PARTITIONING  (Non-IID)
# ============================================================

def partition_non_iid(labels, num_clients, alpha=0.5):
    """Dirichlet-based Non-IID partition."""
    labels       = np.array(labels)
    num_classes  = len(np.unique(labels))
    client_idxs  = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        class_idxs = np.where(labels == c)[0]
        if len(class_idxs) == 0:
            continue
        np.random.shuffle(class_idxs)
        proportions  = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions  = (proportions * len(class_idxs)).astype(int)
        proportions[-1] = len(class_idxs) - proportions[:-1].sum()
        proportions  = np.maximum(proportions, 0)
        split        = np.split(class_idxs, np.cumsum(proportions[:-1]))
        for i, idxs in enumerate(split):
            client_idxs[i].extend(idxs.tolist())

    # ── FIX: remove empty clients ────────────────────────────
    client_idxs = {i: v for i, v in client_idxs.items() if len(v) > 0}
    return client_idxs

num_hospitals  = min(CONFIG["NUM_HOSPITALS"], len(image_paths))
client_indices = partition_non_iid(raw_labels, num_hospitals, alpha=0.5)

print(f"\nHospital Data Distribution ({len(client_indices)} hospitals):")
for hid, idxs in client_indices.items():
    lbs = [int(raw_labels[i]) for i in idxs]
    u, c = np.unique(lbs, return_counts=True)
    dist = {CLASS_NAMES[uu]: cc for uu, cc in zip(u, c)}
    print(f"  Hospital {hid+1}: {len(idxs)} images | {dist}")

# ============================================================
# STEP 4 — CNN-ViT HYBRID MODEL
# ============================================================

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base         = models.efficientnet_b0(
                           weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        return self.pool(self.features(x))   # (B, 1280, 7, 7)

class ViTBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(*([self.norm1(x)] * 3))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class CNNViT(nn.Module):
    def __init__(self, num_classes=4, embed_dim=256, num_heads=8,
                 depth=4, drop=0.1):
        super().__init__()
        self.cnn       = CNNFeatureExtractor()
        self.proj      = nn.Linear(1280, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))
        self.blocks    = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, drop=drop) for _ in range(depth)
        ])
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(128, num_classes),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B      = x.shape[0]
        feat   = self.cnn(x).flatten(2).transpose(1, 2)   # (B,49,1280)
        feat   = self.proj(feat)                           # (B,49,256)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens)[:, 0])

num_classes  = len(CLASS_NAMES)
global_model = CNNViT(num_classes=num_classes).to(DEVICE)
print(f"\nModel: CNN-ViT | Classes: {num_classes} | "
      f"Params: {sum(p.numel() for p in global_model.parameters()):,}")

# ============================================================
# STEP 5 — DP-SGD
# ============================================================

def clip_gradients(model, max_norm):
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def add_dp_noise(model, noise_multiplier, max_norm, num_samples):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=noise_multiplier * max_norm / max(num_samples, 1),
                    size=param.grad.shape,
                ).to(param.grad.device)
                param.grad += noise

def local_train(model, dataset, client_idxs, config):
    """Train one client with DP-SGD. Returns (state_dict, loss, n_samples)."""

    # ── FIX: guard empty client ──────────────────────────────
    if len(client_idxs) == 0:
        return model.state_dict(), 0.0, 0

    model.train()
    subset    = Subset(dataset, client_idxs)
    loader    = DataLoader(subset,
                           batch_size=min(config["BATCH_SIZE"], len(client_idxs)),
                           shuffle=True, num_workers=0, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()
    total_loss, steps = 0.0, 0

    for _ in range(config["LOCAL_EPOCHS"]):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            clip_gradients(model, config["DP_MAX_GRAD_NORM"])
            add_dp_noise(model, config["DP_NOISE_MULT"],
                         config["DP_MAX_GRAD_NORM"], len(client_idxs))
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(steps, 1)
    return model.state_dict(), avg_loss, len(client_idxs)

# ============================================================
# STEP 6 — GSF AGGREGATION
# ============================================================

def extract_update_vector(state_dict):
    return torch.cat([v.float().flatten() for v in state_dict.values()])

def gsf_filter(updates, threshold=2.0):
    norms = np.array([u.norm().item() for u in updates])
    if len(norms) < 3:
        return list(range(len(updates)))
    z     = np.abs(zscore(norms))
    valid = [i for i, zi in enumerate(z) if zi < threshold]
    removed = len(updates) - len(valid)
    if removed:
        print(f"  GSF: {removed} suspicious update(s) filtered")
    return valid if valid else list(range(len(updates)))

def weighted_median_aggregate(state_dicts, weights, valid_idxs):
    filtered_dicts   = [state_dicts[i] for i in valid_idxs]
    filtered_weights = np.array([weights[i] for i in valid_idxs],
                                dtype=np.float32)
    filtered_weights /= filtered_weights.sum()

    new_state = {}
    for key in filtered_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in filtered_dicts])
        w = torch.tensor(filtered_weights,
                         device=stacked.device).view(-1, *([1]*(stacked.dim()-1)))
        sorted_idx  = torch.argsort(stacked, dim=0)
        sorted_vals = torch.gather(stacked, 0, sorted_idx)
        sorted_w    = torch.gather(w.expand_as(stacked), 0, sorted_idx)
        cum_w       = torch.cumsum(sorted_w, dim=0)
        first_above = torch.argmax((cum_w >= 0.5).float(), dim=0, keepdim=True)
        new_state[key] = torch.gather(sorted_vals, 0, first_above).squeeze(0)

    return new_state

# ============================================================
# STEP 7 — FEDERATED TRAINING LOOP
# ============================================================

print("\n" + "="*60)
print("FEDERATED LEARNING — GLOBAL ROUNDS")
print("="*60)

history = {"round": [], "loss": []}

for rnd in range(1, CONFIG["GLOBAL_ROUNDS"] + 1):
    print(f"\n--- Round {rnd}/{CONFIG['GLOBAL_ROUNDS']} ---")
    local_weights, local_losses, local_sizes = [], [], []

    for hid, idxs in client_indices.items():
        local_model = copy.deepcopy(global_model)
        w, loss, n  = local_train(local_model, full_dataset, idxs, CONFIG)

        # ── FIX: skip empty clients ──────────────────────────
        if n == 0:
            print(f"  Hospital {hid+1}: skipped (no samples)")
            continue

        local_weights.append(w)
        local_losses.append(loss)
        local_sizes.append(n)
        print(f"  Hospital {hid+1}: loss={loss:.4f}  samples={n}")

    # ── FIX: skip round if no valid updates ─────────────────
    if not local_weights:
        print("  No valid updates this round — skipping aggregation.")
        continue

    update_vecs = [extract_update_vector(w) for w in local_weights]
    valid_idxs  = gsf_filter(update_vecs, CONFIG["GSF_THRESHOLD"])
    new_global  = weighted_median_aggregate(local_weights, local_sizes, valid_idxs)
    global_model.load_state_dict(new_global)

    avg_loss = np.mean([local_losses[i] for i in valid_idxs])
    history["round"].append(rnd)
    history["loss"].append(avg_loss)
    print(f"  Aggregated Loss : {avg_loss:.4f}")

# ============================================================
# STEP 8 — EVALUATE GLOBAL MODEL
# ============================================================

print("\n" + "="*60)
print("GLOBAL MODEL EVALUATION")
print("="*60)

all_indices = list(range(len(full_dataset)))

# ── FIX: stratified split only if every class has ≥ 2 samples
lbl_arr    = np.array([int(full_dataset.labels[i]) for i in all_indices])
u, c       = np.unique(lbl_arr, return_counts=True)
can_strat  = all(c >= 2)

try:
    _, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=42,
        stratify=lbl_arr if can_strat else None
    )
except ValueError:
    # Fallback: last 20%
    split    = max(1, int(0.8 * len(all_indices)))
    test_idx = all_indices[split:]

# ── FIX: ensure test set is not empty ───────────────────────
if len(test_idx) == 0:
    test_idx = all_indices[-max(1, len(all_indices)//5):]

test_loader = DataLoader(
    Subset(full_dataset, test_idx),
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False, num_workers=0
)

global_model.eval()
all_preds, all_true, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        probs  = F.softmax(global_model(imgs), dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(labels.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
all_probs = np.array(all_probs)

print(classification_report(all_true, all_preds, target_names=CLASS_NAMES,
                             zero_division=0))

# Confusion matrix
cm_val = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — Global Model")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["SAVE_DIR"], "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# Training loss curve
if history["round"]:
    plt.figure(figsize=(8, 4))
    plt.plot(history["round"], history["loss"], 'o-', color='steelblue')
    plt.title("Federated Training Loss per Round")
    plt.xlabel("Global Round"); plt.ylabel("Average Loss")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["SAVE_DIR"], "training_loss.png"), dpi=150)
    plt.close()
    print("Saved: training_loss.png")

# ============================================================
# STEP 9 — GRAD-CAM++
# ============================================================

class GradCAMPlusPlus:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer     = list(model.cnn.features.children())[-1]
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, out):   self.activations = out.detach()
    def _save_grad(self, _, __, g):    self.gradients   = g[0].detach()

    def __call__(self, img_tensor, class_idx=None):
        self.model.eval()
        t      = img_tensor.unsqueeze(0).to(DEVICE)
        logits = self.model(t)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads  = self.gradients[0]
        acts   = self.activations[0]
        alpha_num   = grads ** 2
        alpha_denom = (2 * grads**2
                       + (acts * grads**3).sum(dim=(1,2), keepdim=True)
                       + 1e-7)
        alpha   = alpha_num / alpha_denom
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))
        cam     = F.relu((weights[:, None, None] * acts).sum(dim=0))
        cam     = (cam - cam.min()) / (cam.max() + 1e-7)
        return cam.cpu().numpy(), class_idx

    def remove_hooks(self):
        self._fwd.remove(); self._bwd.remove()

def overlay_heatmap(img_np, cam, alpha=0.5):
    h, w    = img_np.shape[:2]
    cam_r   = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(alpha * heatmap + (1 - alpha) * img_np, 0, 1)

# Use first valid image
sample_path  = image_paths[0]
sample_label = int(raw_labels[0])
raw_img      = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)
raw_resized  = cv2.resize(raw_img, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])) / 255.0
sample_tensor = preprocess(cv2.imread(sample_path))

gradcam              = GradCAMPlusPlus(global_model)
cam_map, pred_class  = gradcam(sample_tensor)
gradcam.remove_hooks()
overlay_img          = overlay_heatmap(raw_resized, cam_map)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(raw_resized);        axes[0].set_title("Original X-Ray");    axes[0].axis("off")
axes[1].imshow(cam_map, cmap='jet'); axes[1].set_title("Grad-CAM++ Map");   axes[1].axis("off")
axes[2].imshow(overlay_img);        axes[2].set_title(
    f"Overlay\nPred: {CLASS_NAMES[pred_class]}");                           axes[2].axis("off")
plt.suptitle("Grad-CAM++ Explainability", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["SAVE_DIR"], "gradcam_overlay.png"), dpi=150)
plt.close()
print("Saved: gradcam_overlay.png")

# ============================================================
# STEP 10 — SHAP
# ============================================================

print("\nRunning SHAP explainer …")

def model_predict_np(imgs_np):
    global_model.eval()
    t = torch.tensor(imgs_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return F.softmax(global_model(t), dim=1).cpu().numpy()

# ── FIX: ensure background set is not empty ─────────────────
bg_size   = min(10, len(full_dataset))
if bg_size == 0:
    raise ValueError("Dataset is empty — cannot run SHAP.")

bg_indices = np.random.choice(len(full_dataset), bg_size, replace=False)
bg_tensors = torch.stack([full_dataset[int(i)][0]
                           for i in bg_indices]).numpy()

sample_np  = sample_tensor.unsqueeze(0).numpy()
explainer  = shap.KernelExplainer(model_predict_np, bg_tensors)
shap_values= explainer.shap_values(sample_np, nsamples=30, l1_reg="aic")

shap_img   = np.abs(shap_values[pred_class][0]).mean(axis=0)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(raw_resized); plt.title("X-Ray"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(shap_img, cmap='hot')
plt.colorbar(label='|SHAP|')
plt.title(f"SHAP — {CLASS_NAMES[pred_class]}"); plt.axis("off")
plt.suptitle("SHAP Feature Importance", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["SAVE_DIR"], "shap_map.png"), dpi=150)
plt.close()
print("Saved: shap_map.png")

# ============================================================
# STEP 11 — FINAL OUTPUT
# ============================================================

print("\n" + "="*60)
print("FINAL PREDICTION OUTPUT")
print("="*60)

global_model.eval()
with torch.no_grad():
    probs_sample = F.softmax(
        global_model(sample_tensor.unsqueeze(0).to(DEVICE)), dim=1
    ).cpu().numpy()[0]

pred_label = CLASS_NAMES[np.argmax(probs_sample)]
confidence = probs_sample.max() * 100

print(f"\n  Sample Image : {os.path.basename(sample_path)}")
print(f"  True Label   : {CLASS_NAMES[sample_label]}")
print(f"  Predicted    : {pred_label}")
print(f"  Confidence   : {confidence:.2f}%")
print("\n  Class Probabilities:")
for cls, prob in zip(CLASS_NAMES, probs_sample):
    bar = "█" * int(prob * 30)
    print(f"    {cls:<15}: {prob*100:5.2f}%  {bar}")

# Final summary figure
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(raw_resized);         axes[0].set_title("Original X-Ray");    axes[0].axis("off")
axes[1].imshow(overlay_img);         axes[1].set_title("Grad-CAM++ Overlay");axes[1].axis("off")
axes[2].imshow(shap_img, cmap='hot');axes[2].set_title("SHAP Map");          axes[2].axis("off")
colors = ['#e74c3c','#3498db','#2ecc71','#95a5a6']
axes[3].barh(CLASS_NAMES, probs_sample,
             color=colors[:len(CLASS_NAMES)])
axes[3].set_xlim(0, 1)
axes[3].set_title("Class Probabilities")
axes[3].set_xlabel("Confidence")
for i, v in enumerate(probs_sample):
    axes[3].text(v + 0.01, i, f"{v*100:.1f}%", va='center', fontsize=9)

plt.suptitle(
    f"Federated Learning Diagnosis\n"
    f"Prediction: {pred_label}  |  Confidence: {confidence:.2f}%",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["SAVE_DIR"], "final_report.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: final_report.png")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(global_model.state_dict(),
           os.path.join(CONFIG["SAVE_DIR"], "federated_cnn_vit.pth"))
print("Saved: federated_cnn_vit.pth")
print(f"\nAll outputs saved to: {CONFIG['SAVE_DIR']}")
print("Pipeline complete ✓")