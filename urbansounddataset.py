from torch.utils.data import Dataset
import pandas as pd 

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotation_file = pd.read_csv('data\UrbanSounds8K\UrbanSound8K.csv')
        self.audio_dir = pd.read_csv('data\UrbanSounds8K')

    def __len__(self):
        pass 

    def __getitem__(self, index):
        pass 

    