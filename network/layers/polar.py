import torch
import cv2
import numpy as np

class Polar(torch.nn.Module):
    def __init__(self):
        super(Polar, self).__init__()
    
    def forward(self, x):
        with torch.no_grad():
            return self.polar(x)
            
    def polar(self, img):
        B, C, H, W = img.shape
        device = img.device
        img = img.cpu().numpy()
        img = np.transpose(img, (0, 2, 3, 1)) # B, H, W, C
        for b in range(B):
            img[b] = cv2.linearPolar(img[b], (H // 2, W // 2), W / 2, cv2.WARP_FILL_OUTLIERS).reshape(H, W, C)
        img = np.transpose(img, (0, 3, 2, 1)) # B, C, W, H
        return torch.from_numpy(img).to(device)