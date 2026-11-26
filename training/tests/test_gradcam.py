import pytest
from PIL import Image

from training.gradcam import activation_inside, object_box_in_crop, overlay


def test_activation_inside_measures_heatmap_share():
    import numpy as np
    cam = np.zeros((10, 10))
    cam[0:5, 0:5] = 1.0
    cam[9, 9] = 1.0
    assert activation_inside(cam, [(0.0, 0.0, 0.5, 0.5)]) == pytest.approx(25 / 26)
    assert activation_inside(cam, [(0.5, 0.5, 0.5, 0.5)]) == pytest.approx(1 / 26)
    assert activation_inside(np.zeros((4, 4)), [(0.0, 0.0, 1.0, 1.0)]) is None


def test_object_box_in_crop_accounts_for_margin():
    x, y, w, h = object_box_in_crop((0.2, 0.2, 0.5, 0.5), 1000, 1000, 0.1)
    assert x == pytest.approx(50 / 600) and w == pytest.approx(500 / 600)
    x, y, w, h = object_box_in_crop((0.0, 0.0, 0.5, 0.5), 1000, 1000, 0.1)
    assert x == 0.0 and w == pytest.approx(500 / 550)


def test_grad_cam_highlights_discriminative_region():
    torch = pytest.importorskip("torch")
    from torch import nn
    from training.gradcam import grad_cam

    class Backbone(nn.Module):
        def forward_features(self, x):
            return x

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.attention = Backbone(), nn.Identity()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(2, 10)
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.bias.zero_()
                self.head.weight[3, 0] = 5.0

    tensor = torch.zeros((1, 2, 8, 8))
    tensor[0, 0, :4, :4] = 1.0
    tensor[0, 1] = 1.0
    cam, target, probabilities = grad_cam(Tiny(), tensor)
    assert target == 3 and len(probabilities) == 10
    assert cam.shape == (8, 8) and cam.max() == pytest.approx(1.0)
    assert activation_inside(cam, [(0.0, 0.0, 0.5, 0.5)]) == pytest.approx(1.0)


def test_overlay_keeps_image_size():
    import numpy as np
    image = Image.new("RGB", (40, 30), (0, 0, 0))
    result = overlay(image, np.ones((4, 4)))
    assert result.size == (40, 30) and result.getpixel((0, 0))[0] > 0
