import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from mpl_toolkits.axes_grid1 import ImageGrid
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms, utils
from PIL import Image

from torchvision.models import resnet18, ResNet18_Weights


from configs import ConfigReader
from utils import PathUtil
from my_dataset import myDataset


# -----------------------------------------------------------------------------

def load_data(
    in_path: str, 
    breeds_file: str,
    labels_file: str, 
    topn: int) -> DataFrame:
    
    # Load breeds
    breeds_path = os.path.join(in_path, breeds_file)
    if not os.path.exists(breeds_path):
        raise FileNotFoundError("File '{breeds_path}' not found.")
    
    breeds_df = pd.read_csv(breeds_path)
    breeds_df = breeds_df.nlargest(topn, 'breed_count')
    
    # Load labels
    # Load breeds
    labels_path = os.path.join(in_path, labels_file)
    if not os.path.exists(labels_path):
        raise FileNotFoundError("File '{labels_path}' not found.")
    
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[labels_df['breed_id'].isin(breeds_df['breed_id'])]
    
    return breeds_df, labels_df

# -----------------------------------------------------------------------------

def create_breed_dicts(in_df: DataFrame):
    """
    Create forward and reverse breed dictionaries.
    """
    breed_to_id = {row['breed']: row['breed_id'] for _, row in in_df.iterrows()}
    id_to_breed = {row['breed_id']: row['breed'] for _, row in in_df.iterrows()}

    return breed_to_id, id_to_breed

# -----------------------------------------------------------------------------

def get_image_path(id: str, train_path: str) -> str:
    img_path = os.path.join(train_path, f"{id}.jpg")
    return img_path if os.path.exists(img_path) else None

# -----------------------------------------------------------------------------

def prepare_data(
    in_df: DataFrame, 
    image_path: str, 
    train_ratio: float=0.8):
    
    # Add image path column
    out_df = in_df.copy()
    out_df['image'] = out_df['id'].apply(
        lambda x: get_image_path(x, image_path)
    )
    
    # Keep only valid data
    out_df = out_df.dropna(subset=['image'])
    
    # Split into train and valid sets
    # out_df = out_df[['image', 'breed_id']]
    out_df = out_df[['id', 'breed_id']]
    
    train_df, valid_df = train_test_split(
        out_df, 
        train_size=train_ratio, 
        random_state=42, 
        stratify=in_df['breed_id']
    )
    
    return train_df, valid_df

# -----------------------------------------------------------------------------

def train_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):
    
    use_gpu = torch.cuda.is_available()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    dataset_sizes = {
        'train': len(dataloders['train'].dataset), 
        'valid': len(dataloders['valid'].dataset)
    }

    for epoch in range(num_epochs):
        
        for phase in ['train', 'valid']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloders[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistic
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()                
            
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 
                'valid loss: {:.4f} acc: {:.4f}'.format(
                epoch+1, num_epochs,
                train_epoch_loss, train_epoch_acc, 
                valid_epoch_loss, valid_epoch_acc))
                    
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, best_acc                    
                    
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    root_path = PathUtil.get_root_path()

    # Config Reader
    resources_path = PathUtil.get_resources_path()
    config_file = os.path.join(resources_path, "application.yaml")
    config_reader = ConfigReader(config_file)

    # Preprocess
    train_config = config_reader.get_config('train')
    data_path   = os.path.join(root_path, train_config['data']['path'])
    breeds_file = train_config['data']['breeds_file']
    labels_file = train_config['data']['labels_file']
    
    num_breed_classes = train_config['num_breed_classes']
    num_epochs = train_config['num_epochs']
    
    # Load data 
    df_breeds, df_labels = load_data(
        data_path, 
        breeds_file, 
        labels_file, 
        num_breed_classes)
    print(f"\nBreed data count: {len(df_breeds)}")
    print(f"\nLabel data count: {len(df_labels)}")
    
    breed_to_id, id_to_breed = create_breed_dicts(df_breeds)
    print(f"\nBreed dict (fw):") 
    print(json.dumps(breed_to_id, ensure_ascii=False, indent=2))
    print(f"\nBreed dict (rv):")
    print(json.dumps(breed_to_id, ensure_ascii=False, indent=2))

    
    # Prepare Data
    image_path = train_config['image']['path']
    
    df_train, df_valid = prepare_data(df_labels, image_path)
    print(f"\nTrain data count: {len(df_labels)}")
    print(f"Train set size: {len(df_train)}")
    print(f"Valid set size: {len(df_valid)}")
    
    # Normalize
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    )

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_set = myDataset(df_train, image_path, transform=transform)
    valid_set = myDataset(df_valid, image_path, transform=transform)
    print('\nTrainSet size: ', len(train_set))
    print('ValidSet size: ', len(valid_set))
    
    batch_size  = train_config['batch_size']
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    print('\nTrainLoader size: ', len(trainLoader))
    print('ValidLoader size: ', len(validLoader))
    
    # Train
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(device)
    
    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用 ResNet18 預訓練權重
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # LoadPreTrained = train_config['load_pre_trained']
    # if LoadPreTrained:
    #     f_abspath = os.path.join(PreTranPath, PreTranModel)
    #     pre_model_wts = torch.load(f_abspath)
    #     model.load_state_dict(pre_model_wts)

    # freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # new final layer with 16 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_breed_classes)

    if use_gpu: model = model.cuda()
    # model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    dataLoaders = { 
        'train':trainLoader, 
        'valid':validLoader 
    }
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("\nOptimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
        
    start_time = time.time()
    model, best_acc = train_model(
        dataLoaders, 
        model, 
        criterion, 
        optimizer, 
        exp_lr_scheduler, 
        num_epochs=num_epochs)
    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))