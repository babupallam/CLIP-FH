import torch
import os

def cache_image_features(clip_model, dataloader, device, model_name="vitb16", cache_dir="cached_feats"):
    os.makedirs(cache_dir, exist_ok=True)
    feat_path = os.path.join(cache_dir, f"{model_name}_image_feats.pt")
    label_path = os.path.join(cache_dir, f"{model_name}_labels.pt")

    # Check for valid existing cache
    if os.path.exists(feat_path) and os.path.exists(label_path):
        try:
            cached_feats = torch.load(feat_path)
            cached_labels = torch.load(label_path)

            # Automatically invalidate cache if shape mismatch
            if cached_feats.shape[0] != len(dataloader.dataset):
                print(f"[Cache] Invalidating {model_name} cache (data length mismatch)")
            elif cached_feats.shape[1] != clip_model.visual.output_dim:
                print(f"[Cache] Invalidating {model_name} cache (dim mismatch)")
            else:
                print(f"[Cache] Loaded valid features for {model_name}")
                return cached_feats, cached_labels
        except Exception as e:
            print(f"[Cache] Failed to load cache for {model_name}: {e}")

    # Rebuild cache
    print(f"[Cache] Generating image features for {model_name}")
    clip_model.eval()
    image_features, labels = [], []

    with torch.no_grad():
        for images, label_batch in dataloader:
            images, label_batch = images.to(device), label_batch.to(device)
            feats = clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features.append(feats.cpu())
            labels.append(label_batch.cpu())

    image_features = torch.cat(image_features)
    labels = torch.cat(labels)

    torch.save(image_features, feat_path)
    torch.save(labels, label_path)
    print(f"[Cache] Saved features  {feat_path}")

    return image_features, labels
