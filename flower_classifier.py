import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import util
import pdb

class FlowerClassifier:
    def __init__(self, arch, hidden_units, gpu):
        self.n_classes = 102
        self.arch = arch
        self.hidden_units = hidden_units
        self.model = self.build_model()
        print(self.model)
        self.class_to_idx = None
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.final_epoch = None
        self.train_losses = []
        self.test_losses = []
        
    def build_model(self):
        # get pretrained model
        model = models.__dict__[self.arch](pretrained=True)
        
        # freeze parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # attach final layer
        if self.arch == 'vgg11':
            model.classifier = self.build_final_layer(25088)
        elif self.arch == 'alexnet':
            model.classifier = self.build_final_layer(9216)
        else:
            raise Exception('Invalid architecture')
            
        return model
            
    def build_final_layer(self, input_layers):
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(input_layers, self.hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(self.hidden_units, self.n_classes)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        return classifier
    
    def train(self, data_dir, epochs, learning_rate):
        image_datasets, dataloaders, class_to_idx  = self.load_data(data_dir)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        
        # gpu or cpu
        self.model.to(self.device)
        
        # start training
        train_losses = []
        test_losses = []
        for e in range(epochs):
            running_train_loss = 0
            self.model.train()
            for images, labels in dataloaders['train']:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # get log probs
                log_ps = self.model.forward(images)

                # get loss
                loss = criterion(log_ps, labels)
                running_train_loss += loss.item()
        #         print(f'running_train_loss: {running_train_loss}')

                # back propagation
                loss.backward()

                # adjust weights
                optimizer.step()

            else:
                self.model.eval()
                running_test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in dataloaders['test']:
                        images, labels = images.to(self.device), labels.to(self.device)

                        # get log probs
                        log_ps = self.model.forward(images)

                        # get loss
                        test_loss = criterion(log_ps, labels)
                        running_test_loss += test_loss.item()
        #                 print(f'running_test_loss: {running_test_loss}')

                        # turn log probs into real probs
                        ps = torch.exp(log_ps)

                        # calc accuracy
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            n_test_batches = len(dataloaders['test'])
            n_train_batches = len(dataloaders['train'])

            epoch_train_loss = running_train_loss / n_train_batches
            epoch_test_loss  = running_test_loss / n_test_batches

            train_losses.append(epoch_train_loss)
            test_losses.append(epoch_test_loss)

            print(f'Epoch: {e+1}/{epochs}',
                  f'Training Loss {epoch_train_loss:{0}.{4}}',
                  f'Validation Loss {epoch_test_loss:{0}.{4}}',
                  f'Accuracy {(accuracy / n_test_batches):{0}.{4}}'
                 )
        
        #return e+1, train_losses, test_losses
        self.final_epoch = e+1
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.class_to_idx = class_to_idx
        
    def load_data(self, data_dir):
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]),
            'test': transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        }
        data_transforms['valid'] = data_transforms['test']

        # Load the datasets with ImageFolder
        image_datasets = {
            'train': datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train']),
            'test': datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test']),
            'valid': datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid'])
        } 

        # Using the image datasets and the trainforms, define the dataloaders
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
            'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
            'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
        }
        
        return image_datasets, dataloaders, image_datasets['train'].class_to_idx
    
    def predict(self, image_path, top_k, cat_to_name_path):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        self.model.to(self.device)

        im = Image.open(image_path)
        im_tensor = util.process_image(im)
    #     print(im_tensor.shape)

        im_tensor.unsqueeze_(0)
#         print(im_tensor.shape)

#         log_ps = self.model.forward(im_tensor.float())
        im_tensor = im_tensor.to(self.device)
    
        # get probabilities without gradients
        with torch.no_grad():
            log_ps = self.model.forward(im_tensor.float())
        ps = torch.exp(log_ps)

        # get top probabilities
        top_probs, top_indexes = ps.topk(top_k)
    #     print(f'top_probs: {top_probs}') 
    #     print(f'top_indexes: {top_indexes}')      

        # flatten
        top_probs = top_probs.cpu().numpy().reshape(top_k,)
    #     print(f'top_probs: {top_probs}')

        top_indexes = top_indexes.cpu().numpy().reshape(top_k,)
    #     print(f'top_indexes: {top_indexes}')

        #print(f'model.class_to_idx: {model.class_to_idx}')

        idx_to_class = {v: str(k) for k, v in self.class_to_idx.items()}
        #print(f'idx_to_class: {idx_to_class}')

        top_classes = []
        for tc in top_indexes: top_classes.append(idx_to_class[tc])
            
        #[idx_to_class[tc] for tc in top_indexes.cpu().numpy()]
        
        # get class names
        class_to_name = util.read_json_file(cat_to_name_path)
        
        return top_probs, [class_to_name[tc] for tc in top_classes]
   
        
        
        
    
        