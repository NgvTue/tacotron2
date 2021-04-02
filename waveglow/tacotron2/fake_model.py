from layers import LinearNorm, ConvNorm
import torch
from torch import nn as nn

import numpy as np
from typing import List, Optional, Set

class Tacotron2Encoder(nn.Module):
    def __init__(self, num_symbol):    


        self.text_embedding = nn.Embedding(num_embeddings=num_symbol, embedding_dim=512)

        convs = []
        for i in range(3):
            convs.append(
                ConvNorm(
                    512,512,kernel_size=5, stride=1,padding=2,bias=False
                )
            )
            convs.append(
                nn.BatchNorm1d(
                    512
                )

            )
            convs.append(
                nn.ReLU()
            )

