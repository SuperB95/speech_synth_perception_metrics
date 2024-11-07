import os
import ffmpeg
from pathlib import Path
import shutil
import h5py
import torch
import numpy as np
import sounddevice as sd
import ustool.ustools.voice_activity_detection as vad
from scipy.io import wavfile
import WaveGlow_functions
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from PIL import Image


class AudioProcessor:
    # recreating the input filestructure with all sound converted to .wav
    @staticmethod
    def path_to_wav(input_dir, output_dir):
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Recursively walk through the input directory
        for root, dirs, files in os.walk(input_dir):
            # Create the corresponding output directory structure
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            for file in files:
                input_file_path = os.path.join(root, file)
                # Only process audio files (ogg, mp3, etc.)
                if file.endswith(('.ogg', '.mp3', '.flac', '.aac', '.m4a')):
                    # Create the output .wav filename with the same name
                    output_file_path = os.path.join(output_subdir, Path(file).stem + '.wav')
                    # Convert the file to .wav using ffmpeg
                    try:
                        print(f"Converting {input_file_path} to {output_file_path}")
                        ffmpeg.input(input_file_path).output(output_file_path).run(quiet=True)
                    except ffmpeg.Error as e:
                        print(f"Error converting {input_file_path}: {e}")
                else:
                    # Copy non-audio files directly to the new path
                    shutil.copy(input_file_path, os.path.join(output_subdir, file))

    @staticmethod
    def perform_vad(file_path, target_sr=22050):
        """
        Perform voice activity detection and separate speech from silence.

        Parameters:
        file_path (str): Path to the audio file.
        target_sr (int): Target sampling rate for the audio file.

        Returns:
        np.ndarray: Speech-only waveform.
        """
        # Load and resample the audio file
        wav, wav_sr = librosa.load(file_path, sr=48000)
        wav = librosa.resample(wav, orig_sr=48000, target_sr=target_sr, res_type='kaiser_best')

        # Perform voice activity detection
        time_segments = vad.detect_voice_activity(wav, target_sr)
        _, speech = vad.separate_silence_and_speech(wav, target_sr, time_segments)

        return speech

    @staticmethod
    def wav_to_mel(wav_data, target_sr=22050, n_melspec=80):
        """
        Convert waveform data to a mel spectrogram.

        Parameters:
        wav_data (np.ndarray): The speech-only waveform.
        target_sr (int): Target sampling rate for mel spectrogram.
        n_melspec (int): Number of mel spectrogram channels.

        Returns:
        np.ndarray: Mel spectrogram data.
        """
        # Save the processed speech data temporarily
        temp_wav = "temp_speech.wav"
        wavfile.write(filename=temp_wav, rate=target_sr, data=wav_data)

        # Mel spectrogram extraction
        stft = WaveGlow_functions.TacotronSTFT(
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=n_melspec,
            sampling_rate=target_sr,
            mel_fmin=0,
            mel_fmax=8000
        )
        mel_data = WaveGlow_functions.get_mel(temp_wav, stft)

        # Remove the temporary file
        os.remove(temp_wav)

        # Convert mel data to numpy
        mel_data = mel_data.data.numpy()
        return mel_data

    @staticmethod
    def create_dataset(base_dir='media/voice_samples', split_ratios=(0.7, 0.15, 0.15), target_sr=22050,
                       n_melspec=80, h5_file='mel_dataset.h5'):
        with h5py.File(h5_file, 'w') as h5f:
            train_group = h5f.create_group("train")
            val_group = h5f.create_group("validation")
            test_group = h5f.create_group("test")

            # Traverse female and male folders
            for gender in ['female', 'male']:
                gender_path = os.path.join(base_dir, gender)

                for subject in os.listdir(gender_path):
                    subject_path = os.path.join(gender_path, subject)
                    if not os.path.isdir(subject_path):
                        continue

                    # List all audio files for the subject
                    audio_files = [os.path.join(subject_path, f) for f in os.listdir(subject_path) if
                                   f.endswith('.wav')]
                    n_samples = len(audio_files)

                    # Calculate sizes for splits
                    train_size = int(n_samples * split_ratios[0])
                    val_size = int(n_samples * split_ratios[1])

                    # Split files into training, validation, and testing
                    train_files = audio_files[:train_size]
                    val_files = audio_files[train_size:train_size + val_size]
                    test_files = audio_files[train_size + val_size:]

                    # Process and save each split
                    AudioProcessor._process_split(train_files, train_group, gender, subject)
                    AudioProcessor._process_split(val_files, val_group, gender, subject)
                    AudioProcessor._process_split(test_files, test_group, gender, subject)

    @staticmethod
    def _process_split(audio_files, group, gender, subject):
        for i, audio_file in enumerate(audio_files):
            audio_file = AudioProcessor.perform_vad(audio_file)
            mel_data = AudioProcessor.wav_to_mel(audio_file)

            # Create a unique path for the mel spectrogram
            mel_path = f"{gender}/{subject}/mel_{i}"
            # Create a dataset for mel spectrogram
            group.create_dataset(mel_path, data=mel_data, compression='gzip')

            # Store audio data in the HDF5 file
            audio_dataset_path = f"{gender}/{subject}/audio_{i}"
            audio_data = AudioProcessor.normalize_audio(audio_file)
            group.create_dataset(audio_dataset_path, data=audio_data, compression='gzip')

    @staticmethod
    def load_audio(audio_file, target_sr=22050):
        """
        Load an audio file and return the waveform as a NumPy array.

        Parameters:
        audio_file (str): Path to the audio file to be loaded.
        target_sr (int): Target sampling rate (default is 22050).

        Returns:
        np.ndarray: Normalized audio waveform.
        """
        # Load the audio file with librosa
        audio_data, sr = librosa.load(audio_file, sr=target_sr)

        # Normalize the audio data to the range [-1, 1]
        audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize

        return audio_data

    @staticmethod
    def normalize_audio(audio_data):
        """
        Normalize the audio waveform data to the range [-1, 1].

        Parameters:
        audio_data (np.ndarray): The audio waveform data.

        Returns:
        np.ndarray: Normalized audio waveform.
        """
        # Avoid division by zero in case of silence (all zeros)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        return audio_data

    # @staticmethod
    # def create_dataset(base_dir='media/voice_samples', split_ratios=(0.7, 0.15, 0.15), target_sr=22050, n_melspec=80, h5_file='mel_dataset.h5'):
    #     with h5py.File(h5_file, 'w') as h5f:
    #         train_group = h5f.create_group("train")
    #         val_group = h5f.create_group("validation")
    #         test_group = h5f.create_group("test")
    #
    #         # Traverse female and male folders
    #         for gender in ['female', 'male']:
    #             gender_path = os.path.join(base_dir, gender)
    #
    #             for subject in os.listdir(gender_path):
    #                 subject_path = os.path.join(gender_path, subject)
    #                 if not os.path.isdir(subject_path):
    #                     continue
    #
    #                 # List all audio files for the subject
    #                 audio_files = [os.path.join(subject_path, f) for f in os.listdir(subject_path) if f.endswith('.wav')]
    #                 n_samples = len(audio_files)
    #
    #                 # Calculate sizes for splits
    #                 train_size = int(n_samples * split_ratios[0])
    #                 val_size = int(n_samples * split_ratios[1])
    #
    #                 # Split files into training, validation, and testing
    #                 train_files = audio_files[:train_size]
    #                 val_files = audio_files[train_size:train_size + val_size]
    #                 test_files = audio_files[train_size + val_size:]
    #
    #                 # Process and save each split
    #                 AudioProcessor._process_split(train_files, train_group, gender, subject)
    #                 AudioProcessor._process_split(val_files, val_group, gender, subject)
    #                 AudioProcessor._process_split(test_files, test_group, gender, subject)
    #
    # @staticmethod
    # def _process_split(files, group, gender, subject):
    #     for i, file_path in enumerate(files):
    #         mel_data = AudioProcessor.wav_to_mel(file_path)
    #         dset = group.create_dataset(f"{gender}/{subject}/mel_{i}", data=mel_data)
    #         dset.attrs['gender'] = gender
    #         dset.attrs['subject'] = subject

    @staticmethod
    def plot_mel_spectrogram(mel_data, sr=22050, hop_length=256, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.imshow(mel_data, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.xlabel("Time (frames)")
        plt.ylabel("Mel Frequency")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='png')
            print(f"Mel spectrogram saved to {save_path}")
        plt.show()

    @staticmethod
    def reconstruct_mel(mel_data, sr=22050, target_sr=48000, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000):
        """
        Reconstruct an audio waveform from a Mel spectrogram and resample it.

        Parameters:
        - mel_data: The input Mel spectrogram to reconstruct from (in numpy array format).
        - sr: Sampling rate.
        - target_sr: The target sampling rate for resampling.
        - n_fft: Number of FFT components.
        - hop_length: The hop length for STFT.
        - win_length: The window length for STFT.
        - fmin: Minimum frequency.
        - fmax: Maximum frequency.

        Returns:
        - reconstructed_audio: Resampled waveform.
        """
        # Convert Mel spectrogram to magnitude spectrogram
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel_data.shape[0], fmin=fmin, fmax=fmax)
        mag_spec = np.dot(np.linalg.pinv(mel_basis), mel_data)

        # Reconstruct the waveform from the magnitude spectrogram using the Griffin-Lim algorithm
        reconstructed_audio = librosa.feature.inverse.griffinlim(mag_spec, n_iter=32, hop_length=hop_length,
                                                                 win_length=win_length)

        # Resample the audio
        reconstructed_audio = librosa.resample(reconstructed_audio, orig_sr=sr, target_sr=target_sr)

        return reconstructed_audio

    @staticmethod
    def save_reconstructed_audio(reconstructed_audio, output_path, sr=22050):
        """
        Save the reconstructed audio waveform to a file.

        Parameters:
        - reconstructed_audio: Reconstructed audio waveform.
        - output_path: The path where the audio will be saved.
        - sr: Sampling rate.
        """
        sf.write(output_path, reconstructed_audio, sr)

    @staticmethod
    def play_audio(audio, sr=22050):
        """
        Play audio directly without saving it.

        Parameters:
        - reconstructed_audio: The audio data to play.
        - sr: The sample rate of the audio.
        """
        sd.play(audio, sr)
        sd.wait()
