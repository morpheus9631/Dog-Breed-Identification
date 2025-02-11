import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class myDataset(Dataset):

    def __init__(self, 
        in_df,
        image_path, 
        transform=None,
    ):
        self.data = in_df
        self.image_path = image_path
        self.transform = transform
        self.len = len(self.data)

    # -------------------------------------------------------------------------

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_path = join(self.image_path, row['id'] + '.jpg')
        img_pil = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = img_pil
        
        lbl = int(row['breed_id'])
        
        return [img, lbl]

    # -------------------------------------------------------------------------

    def __len__(self):
        return self.len
