"""
Grad-CAM — Görüntü modeli için açıklanabilirlik.
Modelin hangi bölgeye bakarak karar verdiğini ısı haritası olarak gösterir.
"""
import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        if target_layer is None:
            # MobileNetV2 son convolutional katman
            target_layer = model.backbone.features[-1]
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        min_val, max_val = cam.min(), cam.max()
        if max_val - min_val > 1e-8:
            cam = (cam - min_val) / (max_val - min_val)
        else:
            cam = np.zeros_like(cam)
        return cam

    @staticmethod
    def overlay(original_image_np, cam_map, alpha=0.5):
        """
        original_image_np: (H, W, 3) uint8 RGB
        cam_map: (224, 224) float [0,1]
        """
        h, w = original_image_np.shape[:2]
        cam_resized = cv2.resize(cam_map, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = np.uint8(alpha * heatmap + (1 - alpha) * original_image_np)
        return blended
