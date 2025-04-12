import torch

def cache_image_features(clip_model, dataloader, device):
    clip_model.eval()
    image_features, labels = [], []

    with torch.no_grad():
        for images, label_batch in dataloader:
            images, label_batch = images.to(device), label_batch.to(device)
            feats = clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features.append(feats.cpu())
            labels.append(label_batch.cpu())

    return torch.cat(image_features), torch.cat(labels)
