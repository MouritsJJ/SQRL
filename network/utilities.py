import math
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from constants import *

def rotate(img, r_min, r_max):
    pic = img.rotate(random.randint(r_min, r_max))

    (H, W, C) = np.array(img).shape
    pad = math.sqrt(math.pow(H // 2, 2) + math.pow(W // 2, 2))
    pad = (int)(2 * pad - H) // 2 + 1

    pic = np.pad(np.array(img), ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    pic = Image.fromarray(pic).rotate(32)
    pic = np.array(pic)[pad:H+pad, pad:W+pad]
    pic = Image.fromarray(pic)

    return pic

def validation(model, device, criterion, data_validation, p=True):
    model.eval()
    validation_loss, correct, num_data = 0, 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_validation):
            images, labels = images.to(device), labels.to(device)
            result = model(images)
            loss = criterion(result, labels)
            validation_loss += loss.float()
            pred = result.argmax(dim=1, keepdim=True)
            pred = pred.view(pred.size()[0])
            correct += pred.eq(labels.view_as(pred)).sum().item()
            num_data += len(images)
        
    validation_loss /= len(data_validation)
    accuracy = 100. * correct / num_data
    if p: 
        print('Validation Loss: {:.6f} Accuracy: {}/{} ({})'.format( validation_loss, correct, num_data, accuracy))
    return validation_loss, accuracy

def load_data(location):
    trans = []
    if gray: trans += [transforms.Grayscale()]
    if resize: trans += [transforms.Resize(image_size)]
    if pad: trans += [transforms.Pad(padding)]
    trans += [transforms.ToTensor()]
    if gray: trans += [transforms.Normalize(*stats_gray)]
    else: trans += [transforms.Normalize(*stats_rgb)]
    dataset = dset.ImageFolder(root=location, 
                                transform=transforms.Compose(trans
                                ))

    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
