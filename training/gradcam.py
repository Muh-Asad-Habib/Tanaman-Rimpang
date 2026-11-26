"""Grad-CAM for the EfficientNetV2-B0 + CBAM classifier.

The heatmap is computed on the attention-refined feature map (output of CBAM, or of the backbone
when CBAM is disabled), i.e. the last spatial tensor before global pooling. `activation_inside`
quantifies whether the model looks at the rhizome: the share of heatmap mass inside the object box.
"""

from training.evaluation import checked_box, padded_crop


def grad_cam(model, tensor, target=None):
    """Return (cam HxW in [0, 1] at feature resolution, target class, softmax probabilities)."""
    import torch
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        base = model.backbone.forward_features(tensor).detach().requires_grad_(True)
        features = model.attention(base)
        features.retain_grad()
        logits = model.head(model.pool(features).flatten(1))
        if logits.shape[0] != 1:
            raise ValueError("Grad-CAM explains one image at a time.")
        probabilities = logits.softmax(1)[0].detach()
        target = int(probabilities.argmax()) if target is None else int(target)
        logits[0, target].backward()
    weights = features.grad[0].mean((1, 2), keepdim=True)
    cam = torch.relu((weights * features[0].detach()).sum(0))
    peak = cam.max()
    cam = cam / peak if peak > 0 else cam
    return cam.cpu().numpy(), target, probabilities.cpu().tolist()


def resize_cam(cam, width, height):
    import numpy as np
    from PIL import Image
    scaled = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(scaled, dtype=np.float64) / 255


def activation_inside(cam, boxes):
    """Fraction of heatmap mass inside the union of normalized [x, y, w, h] boxes; None if the map is empty."""
    import numpy as np
    cam = np.asarray(cam, dtype=np.float64)
    total = cam.sum()
    if total <= 0:
        return None
    height, width = cam.shape
    mask = np.zeros_like(cam, dtype=bool)
    for box in boxes:
        x, y, w, h = checked_box(box)
        mask[int(y * height):max(int(y * height) + 1, round((y + h) * height)),
             int(x * width):max(int(x * width) + 1, round((x + w) * width))] = True
    return float(cam[mask].sum() / total)


def object_box_in_crop(box, width, height, margin=0.1):
    """Where the annotated object sits inside its padded classifier crop, as a normalized box."""
    x, y, w, h = checked_box(box)
    left, top, right, bottom = padded_crop(box, width, height, margin)
    crop_width, crop_height = right - left, bottom - top
    inner_left = max(0.0, (x * width - left) / crop_width)
    inner_top = max(0.0, (y * height - top) / crop_height)
    inner_right = min(1.0, ((x + w) * width - left) / crop_width)
    inner_bottom = min(1.0, ((y + h) * height - top) / crop_height)
    return inner_left, inner_top, inner_right - inner_left, inner_bottom - inner_top


def overlay(image, cam, alpha=0.45):
    """Blend a red-yellow heatmap over the image (no matplotlib dependency)."""
    import numpy as np
    from PIL import Image
    heat = resize_cam(cam, *image.size)
    colors = np.stack([np.ones_like(heat), heat, np.zeros_like(heat)], axis=-1) * 255
    base = np.asarray(image.convert("RGB"), dtype=np.float64)
    weight = (alpha * heat)[..., None]
    return Image.fromarray(np.clip(base * (1 - weight) + colors * weight, 0, 255).astype(np.uint8), "RGB")
