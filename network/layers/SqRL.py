import torch
import numpy as np

class SqRL(torch.nn.Module):
    def __init__(self):
        super(SqRL, self).__init__()
    
    def forward(self, x):
        with torch.no_grad():
            return SquareRotationalLayer(x)
    
    # Square ratotation that keeps correlation in corners
    # Uses reflection in the end
    def SquareRotationalLayer(self, img):
        device = img.device
        # Resize to minimum 32 x 32 - Assuming img is squared and height % 2 = 0
        h = img.shape[2]
        assert h % 2 == 0
        if h < 32:
            pad = (32 - h) // 2
            paddings = ((0, 0), (0, 0), (pad, pad), (pad, pad))
            img = np.pad(img, paddings, mode='symmetric')

        # Run SqRL algorithm
        (B, C, H, W) = img.shape
        res = torch.zeros((B, C, int(np.ceil(H / 2)), H * 4), dtype=img.dtype).to(device)
        lmid = int(np.floor((H - 1) / 2))
        for i in range(lmid, -1, -1):
            dif = lmid - i
            el = (2 * dif) if H % 2 == 1 else (2 * dif + 1)
            # top row
            res[:, :, dif, 0:1*i]               = img[:, :, i, i].repeat(1*i).reshape(B, C, 1*i)
            res[:, :, dif, 1*i:1*i+el]          = img[:, :, i, i:i+el]
            # right column
            res[:, :, dif, 1*i+el:3*i+el]       = img[:, :, i, W - i - 1].repeat(2*i).reshape(B, C, 2*i)
            res[:, :, dif, 3*i+el:3*i+el*2]     = img[:, :, i:i+el, W - i - 1]
            # bottom row
            res[:, :, dif, 3*i+el*2:5*i+el*2]   = img[:, :, i+el, i+el].repeat(2*i).reshape(B, C, 2*i)
            if el != 0: res[:, :, dif, 5*i+2*el:5*i+3*el]   = np.fliplr(img[:, :, i + el, i+1:i+1+el].transpose(0, 2, 1)).transpose(0, 2, 1)
            # left column
            res[:, :, dif, 5*i+3*el:7*i+3*el]   = img[:, :, i+el, i].repeat(2*i).reshape(B, C, 2*i)
            if el != 0: res[:, :, dif, 7*i+3*el:7*i+4*el]   = np.fliplr(img[:, :, i+1:i+1+el, i].transpose(0, 2, 1)).transpose(0, 2, 1)
            # Add missing initial corner
            res[:, :, dif, 7*i+4*el:8*i+4*el] = img[:, :, i, i].repeat(1*i).reshape(B, C, 1*i)
        
        res[:, :, :, (H-1)*4:H*4] = res[:, :, :, 0:4]
        
        return res
