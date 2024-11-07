import h5py
import torch
from torch.utils.data import Dataset


class MelDataset(Dataset):
    def __init__(self, file_path, group='train'):
        self.file_path = file_path
        self.group = group

        # Collect all dataset paths (mel spectrograms only)
        self.data_paths = []
        print("mel paths:")
        with h5py.File(self.file_path, 'r') as h5f:
            group_path = h5f[self.group]
            for gender in group_path:
                for subject in group_path[gender]:
                    subject_group = group_path[gender][subject]
                    for mel_name in subject_group:
                        if "mel" in mel_name:
                            mel_path = f"{self.group}/{gender}/{subject}/{mel_name}"
                            print(mel_path)
                            self.data_paths.append(mel_path)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        mel_path = self.data_paths[idx]

        with h5py.File(self.file_path, 'r') as h5f:
            mel_data = h5f[mel_path][:]

        mel_tensor = torch.tensor(mel_data, dtype=torch.float32)

        audio_path = mel_path.replace('mel', 'audio')
        with h5py.File(self.file_path, 'r') as h5f:
            audio_data = h5f[audio_path][:]

        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        return mel_tensor, audio_tensor

