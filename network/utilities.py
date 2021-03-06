import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from constants import *

# Rotates each image in a batch randomly between angle_min and angle_max
def rotate(X, angle_min, angle_max):
    (H, W) = X.shape[1:3]
    xpad, pad = add_padding(X, H)
    xrot = []
    for x in tqdm(xpad):
        xrot.append(rotate_img(x, angle_min, angle_max))
    return remove_padding(np.array(xrot), H, W, pad)

'''
    Randomly chooses an angle between r_min and r_max
    Rotates the image this angle around the center
'''
def rotate_img(X, r_min, r_max):
    angle = np.random.randint(r_min, r_max)
    return np.array(Image.fromarray((X*255.).astype(np.uint8)).rotate(angle)).astype(np.float32)/255.

# Adds padding to a batch of images - using symmetric padding
def add_padding(X, H):
    # Assuming X is squared
    pad = H / 2 * 2**0.5 * 2
    pad = int(np.ceil((pad - H) / 2))

    paddings = ((0, 0), (pad, pad), (pad, pad)) if len(X.shape) == 3 else ((0, 0), (pad, pad), (pad, pad), (0, 0))

    return np.pad(X, paddings, mode='symmetric'), pad

# Removes padding from a batch of images
def remove_padding(X, H, W, pad):
    return X[:, pad:H+pad, pad:W+pad]

'''
    Validates a model using a data set
    Outputs the validation loss and validation accuracy
'''
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


# Loads the images and the corresponding labels 
def load_data(x_location, y_location, device, shuffle):
    x, y = np.load(x_location).transpose(0, 3, 1, 2), np.load(y_location)
    dataset = torch.utils.data.TensorDataset(torch.tensor(x).to(device), torch.LongTensor(y.ravel()).to(device))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
