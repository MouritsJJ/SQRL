import torch
import math
import time
import numpy as np
from PIL import Image

class SqRL(torch.nn.Module):
    def __init__(self):
        super(SqRL, self).__init__()
    
    def forward(self, x):
        with torch.no_grad():
            return SquareRotationalLayer(x)
    
    # Square ratotation that keeps correlation in corners
    # Uses reflection in the end
    def SquareRotationalLayer(self, img):
        with torch.no_grad():
            (B, C, H, W) = img.shape
            res = torch.zeros((B, C, H // 2, H * 4), dtype=img.dtype).to(img.device)
            # res1 = torch.zeros((B, C, (H+1) // 2, (H+1) * 4), dtype=img.dtype).to(img.device)
            lmid = math.floor((H - 1) / 2)
            for i in range(lmid, -1, -1):
                dif = lmid - i
                el = 2 * dif if H % 2 == 1 else 2 * dif + 1
                # top row
                res[:, :, dif, 0:1*i]               = img[:, :, i, i].repeat(1, 1, 1, 1*i).reshape(B, C, 1*i)
                res[:, :, dif, 1*i:1*i+el]          = img[:, :, i, i:i+el]
                # right column
                res[:, :, dif, 1*i+el:3*i+el]       = img[:, :, i, W - i - 1].repeat(1, 1, 1, 2*i).reshape(B, C, 2*i)
                res[:, :, dif, 3*i+el:3*i+el*2]     = img[:, :, i:i+el, W - i - 1]
                # bottom row
                res[:, :, dif, 3*i+el*2:5*i+el*2]   = img[:, :, i+el, i+el].repeat(1, 1, 1, 2*i).reshape(B, C, 2*i)
                res[:, :, dif, 5*i+2*el:5*i+3*el]   = torch.fliplr(img[:, :, i + el, i+1:i+1+el].transpose(1, 2)).transpose(1, 2)
                # left column
                res[:, :, dif, 5*i+3*el:7*i+3*el]   = img[:, :, i+el, i].repeat(1, 1, 1, 2*i).reshape(B, C, 2*i)
                res[:, :, dif, 7*i+3*el:7*i+4*el]   = torch.fliplr(img[:, :, i+1:i+1+el, i].transpose(1, 2)).transpose(1, 2)
                # Add missing initial corner
                res[:, :, dif, 7*i+4*el:8*i+4*el] = img[:, :, i, i].repeat(1, 1, 1, 1*i).reshape(B, C, 1*i)
            
            # el = 0
            # for i in range((H - 1) * 4):
            #     if i == 0 or i % (H - 1) == 0:
            #         res1[:, :, :, el:el+3] = res[:, :, :, i].repeat(1, 1, 1, 3).reshape(B, C, res.shape[2],3).transpose(2, 3).reshape(B, C, res.shape[2],3)
            #         el += 3
            #     else:
            #         res1[:, :, :, el] = res[:, :, :, i]
            #         el += 1
            res[:, :, :, 4*H-4:4*H+4] = res[:, :, :, 0:8]
            return res