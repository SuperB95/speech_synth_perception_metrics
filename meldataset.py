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
            audio_data = h5f[audio_path][:]  # Load the corresponding audio

        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        # Only return the mel spectrogram for now
        return mel_tensor, audio_tensor

    # def get_audio_for_mel(self, mel_tensor):
    #     # Assuming you have a way to retrieve the corresponding audio for a mel tensor
    #     mel_path = self.get_mel_path(mel_tensor)  # Retrieve the mel file path (based on your dataset structure)
    #     audio_path = mel_path.replace('mel', 'audio')  # Assuming the audio path follows this structure
    #
    #     with h5py.File(self.file_path, 'r') as h5f:
    #         audio_data = h5f[audio_path][:]  # Load the corresponding audio
    #
    #     audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
    #     return audio_tensor

        # def __getitem__(self, idx):
    #     mel_path = self.data_paths[idx]
    #     audio_path = self.get_audio_path(mel_path)  # Implement a method to get the corresponding audio path
    #
    #     with h5py.File(self.file_path, 'r') as h5f:
    #         mel_data = h5f[mel_path][:]
    #         audio_data = h5f[audio_path][:]  # Assuming audio data is also stored in the same HDF5 file
    #
    #     mel_tensor = torch.tensor(mel_data, dtype=torch.float32)
    #     audio_tensor = torch.tensor(audio_data, dtype=torch.float32)  # Convert audio data to tensor
    #
    #     return mel_tensor, audio_tensor  # Return both tensors
    #
    # def get_audio_path(self, mel_path):
    #     # Assuming mel_path is in the form 'train/female/subject_1/mel_0'
    #     audio_path = mel_path.replace('mel', 'audio')  # Adjust this logic as per your data structure
    #     print(audio_path)
    #     return audio_path
