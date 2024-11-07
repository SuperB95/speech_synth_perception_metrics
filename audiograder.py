import argparse
import librosa
import torchmetrics
import torch
from pymcd.mcd import Calculate_MCD
from asteroid_filterbanks import STFTFB, Encoder, transforms
from asteroid.losses import SingleSrcPMSQE


class AudioGrader:
    def __init__(self, original_filepath, generated_filepath):
        self.original_filepath = original_filepath
        self.generated_filepath = generated_filepath

        # Load audio files
        self.y_original, self.sr_original = librosa.load(original_filepath, sr=16000)
        self.y_generated, self.sr_generated = librosa.load(generated_filepath, sr=16000)

        # Ensure both audios have the same length
        min_length = min(len(self.y_original), len(self.y_generated))
        self.y_original = self.y_original[:min_length]
        self.y_generated = self.y_generated[:min_length]

        # Convert audio data to PyTorch tensors
        self.y_original_tensor = torch.tensor(self.y_original)
        self.y_generated_tensor = torch.tensor(self.y_generated)

    def calculate_pesq(self, narrow=False):
        if narrow:
            nb_pesq_metric = torchmetrics.audio.PerceptualEvaluationSpeechQuality(self.sr_original, 'nb')
            pesq = nb_pesq_metric(self.y_generated_tensor, self.y_original_tensor)
        else:
            wb_pesq_metric = torchmetrics.audio.PerceptualEvaluationSpeechQuality(self.sr_original, 'wb')
            pesq = wb_pesq_metric(self.y_generated_tensor, self.y_original_tensor)
        return round(pesq.item(), 4)

    def calculate_stoi(self, extended=False):
        stoi_metric = torchmetrics.audio.ShortTimeObjectiveIntelligibility(self.sr_original, extended=extended)
        score = stoi_metric(self.y_generated_tensor, self.y_original_tensor)
        return round(score.item(), 4)

    def calculate_sisdr(self):
        sisdr_metric = torchmetrics.audio.ScaleInvariantSignalDistortionRatio()
        score = sisdr_metric(self.y_generated_tensor, self.y_original_tensor)
        return round(score.item(), 4)
    def calculate_sdr(self):
        sdr_metric = torchmetrics.audio.SignalDistortionRatio()
        score = sdr_metric(self.y_generated_tensor, self.y_original_tensor)
        return round(score.item(), 4)

    def calculate_pmsqe(self):
        stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256))
        ref_spec = transforms.mag(stft(torch.tensor(self.y_original).unsqueeze(0)))
        est_spec = transforms.mag(stft(torch.tensor(self.y_generated).unsqueeze(0)))
        loss_func = SingleSrcPMSQE()
        score = loss_func(est_spec, ref_spec)
        return round(score.item(), 4)

    def calculate_mcd(self):
        mcd_toolbox = Calculate_MCD(MCD_mode="plain")
        score = mcd_toolbox.calculate_mcd(self.original_filepath, self.generated_filepath)
        return round(score.item(), 4)

    def run_all_metrics(self):
        return {
            "wb_pesq": self.calculate_pesq(),
            "nb_pesq": self.calculate_pesq(narrow=True),
            "stoi": self.calculate_stoi(),
            "estoi": self.calculate_stoi(extended=True),
            "sisdr": self.calculate_sisdr(),
            "sdr": self.calculate_sdr(),
            "pmsqe": self.calculate_pmsqe(),
            "mcd": self.calculate_mcd()
        }
