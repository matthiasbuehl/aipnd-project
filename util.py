import torch
import numpy as np
import pdb
import json

def read_json_file(path):
        with open(path, 'r') as f:
            text = json.load(f)
        
        return text

def save_checkpoint(f_class, path):
    torch.save({
            'epoch': f_class.final_epoch,
            'model_state_dict': f_class.model.state_dict(),
            'class_to_idx': f_class.class_to_idx,
            'train_losses': f_class.train_losses,
            'test_losses': f_class.test_losses
            }, path)
    
def load_checkpoint(f_class, path):
    checkpoint = torch.load(path)
    #print(checkpoint['class_to_idx'])
    #model = models.vgg11(pretrained=True)
    #model.classifier = make_classifier(len(checkpoint['class_to_idx']))
    
    f_class.model.load_state_dict(checkpoint['model_state_dict'])
    f_class.class_to_idx = checkpoint['class_to_idx']
    
    return f_class

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
#     print(f'image.size: {image.size}')
    
    
    # resize
    smaller_size = 256
    width, height = image.size
    smaller = min(width, height)
    
    smaller_width = smaller == image.size[0]
    
    if smaller_width:
        new_height = round(smaller_size * height / width)
        new_width = smaller_size
    else:
        new_width = round(width * smaller_size / height)
        new_height = smaller_size
        
    image = image.resize(size=(new_width, new_height))
#     print(f'image.size: {image.size}')
    
    
    # crop
    crop_size = 244
    left = (new_width - crop_size) / 2
    upper = (new_height - crop_size) / 2
    right = left + crop_size
    lower = upper + crop_size
#     print(f'(left, upper, right, lower): {(left, upper, right, lower)}')
    
    image = image.crop(box=(left, upper, right, lower))
#     print(f'image.size: {image.size}')
    
    
    # normalize
    np_image = np.array(image)
    np_image = np_image / 255
#     print(f'np_image.shape: {np_image.shape}')
#     print(f'np_image[0][0]: {np_image[0][0]}')
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std
#     print(f'np_image[0][0]: {np_image[0][0]}')
    
    np_image = np_image.transpose(2, 0, 1)
#     print(f'np_image.shape: {np_image.shape}')
    
    return torch.from_numpy(np_image)
