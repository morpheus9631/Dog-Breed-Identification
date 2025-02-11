import json
import os
import pandas as pd

from pandas import DataFrame

from configs import ConfigReader
from utils import PathUtil

# -----------------------------------------------------------------------------

def load_input_data(config_reader: ConfigReader) -> DataFrame:
    
    label_config = config_reader.get_config('input')
    label_path = label_config.get('path')
    label_file = label_config.get('file')
    csl_pathfile = os.path.join(label_path, label_file)
    out_df = pd.read_csv(csl_pathfile)
    
    return out_df

# -----------------------------------------------------------------------------

import pandas as pd

def encode_breed(in_df: DataFrame) -> DataFrame:

    # Create a dictionary mapping each breed to a unique ID, starting from 1
    breed_counts = in_df['breed'].value_counts().sort_values(ascending=False)
    breed_mapping = {
        breed: idx for idx, breed in enumerate(breed_counts.index)
    }
    
    # Build a new DataFrame containing breed_id, breed, and breed_count
    out_df = pd.DataFrame({
        'breed_id': breed_counts.index.map(breed_mapping),
        'breed': breed_counts.index,
        'breed_count': breed_counts.values
    })

    return out_df

# -----------------------------------------------------------------------------

def map_image_to_breed(
    input_df: DataFrame, 
    breed_df:  DataFrame
) -> DataFrame:
    
    # Merge df_input with df_breed on the breed column
    out_df = input_df.merge(
        breed_df[['breed_id', 'breed']], on='breed', how='left'
    )

    # Select required columns
    out_df = out_df[['id', 'breed_id']]
    
    return out_df

# -----------------------------------------------------------------------------

def save_to_csv(in_df: DataFrame, out_path: str, out_file: str):
    
    out_pathfile = os.path.join(out_path, out_file)
    in_df.to_csv(out_pathfile, index=False)
    print(f"\nSaved to '{out_pathfile}'")

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    root_path = PathUtil.get_root_path()

    # Config Reader
    resources_path = PathUtil.get_resources_path()
    config_file = os.path.join(resources_path, "application.yaml")
    config_reader = ConfigReader(config_file)

    # Load input data
    input_path = config_reader.get_config('input.path')
    input_file = config_reader.get_config('input.file')

    df_input = load_input_data(config_reader)
    print(f"\nInput data:")
    print(df_input.head())

    # Data preprocess
    preproc_config = config_reader.get_config('preprocess')
    out_path    = os.path.join(root_path, preproc_config['out_path'])
    breeds_file = preproc_config['out_file']['breeds']
    labels_file = preproc_config['out_file']['labels']

    # Encode breed
    df_breed = encode_breed(df_input)
    print(f"\nPreprocess breed id:")
    print(df_breed.head())
    
    save_to_csv(df_breed, out_path, breeds_file)

    # Map image id to breed id
    df_label = map_image_to_breed(df_input, df_breed)
    print(f"\nPreprocess labels:")
    print(df_label.head())
    
    save_to_csv(df_label, out_path, labels_file)
    
