import os
import time
import argparse
import math
from numpy import finfo
from text import sequence_to_text
import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams

def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
if __name__ =='__main__':
    hparams = create_hparams()
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)
    z = train_loader.dataset.get_text("hơn một trăm * nhưng họ chia thành từng nhóm nhỏ #")
    print(z)
    print(train_loader.dataset.text_embedding.symbol2numeric_dict)
    a = z.detach().numpy().tolist()
    reverser_dict = {
        v:k for k,v in train_loader.dataset.text_embedding.symbol2numeric_dict.items()
    }
    a = [reverser_dict[i] for i in a]
    print("".join(a))
    text,mel = train_loader.dataset[0]
    print(mel.shape)
    fig, ax = plt.subplots()
    # M = librosa.feature.melspectrogram(y=y, sr=sr)
    # M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(mel.numpy(), y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel spectrogram display')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig("./sample_mel.png")
    # print(sequence_to_text(z.detach().cpu().numpy().tolist()))