import torch
import numpy as np
from torch import nn as nn

def get_padding(kernel_size, stride, dilation):
    if stride == 1:
        return int((kernel_size*dilation - dilation)/2) 
    else:
        return 0
class Resblock1(nn.Module):
    def __init__(self, hparams, input_size, out_size, kernel_size, dilations):
        super().__init__()
        if not isinstance(kernel_size,list):
            kernel_size = [kernel_size,] * len(dilations)
        layers = []
        for i,k in zip(dilations,kernel_size):
            layers.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        input_size,out_size,kernel_size=k, dilation=i, padding=get_padding(k, 1, i)
                    )
                )
            )
        
        self.convs1 = nn.ModuleList(layers)

        layers = []
        for i,k in zip(dilations, kernel_size):
            layers.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                            input_size,out_size,kernel_size=k , dilation=i, padding=get_padding(k, 1, i)
                        )
                    )
                )
        self.convs2 = nn.ModuleList(layers)
        self.init_weights(self.convs1)
        self.init_weights(self.convs2)


    def init_weights(self, module, mean=0., std=0.01):
        
        if "Conv1" in module.__class__.__name__ :
            module.weight.data.normal_(mean, std)

    def forward(self, inputs):
        # print("here")
        for c1,c2 in zip(self.convs1, self.convs2):
            # print(inputs.size())
            out1 = c1(inputs)
            out1 = nn.functional.leaky_relu(out1, 0.1)
            out2 = c2(out1)
            out2 = nn.functional.leaky_relu(out2, 0.1)
            
            inputs = inputs + out2

        return inputs




class MRFModule(nn.Module):
    def __init__(self, hparams, input_size, out_size):
        super().__init__()
        
        
        kr  = hparams.resblock_kernel_sizes
          
        dr = hparams.resblock_dilation_sizes
        self.kr = len(dr)
        if not isinstance(kr, list):
            kr = [kr,] * self.kr
        layers = []
        for i in range(self.kr):    
            kernel_size = kr[i] # 
            dialotions = dr[i] 
            layers.append(
                Resblock1(
                    hparams,
                    input_size, out_size, kernel_size, dialotions
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        print(inputs.size())
        xs = self.layers[0](inputs)
        for i in range(1,self.kr):
            xs = xs + self.layers[i](inputs)
        
        return xs / self.kr


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(
                80, 512, 7, 1, padding=3 
            )
        )

        self.ups = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        up_sample_rate = [8, 8, 2, 2]
        up_sample_kernel = [16, 16, 4, 4]
        for i in range(len(up_sample_kernel)):
            up_r = up_sample_rate[i]
            up_k  = up_sample_kernel[i]
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        512 // 2**i, 512 // 2**(i+1),
                        up_k,up_r,padding=(up_k - up_r) // 2
                    )
                )
            )
            ch = 512 // 2**(i+1)
            
            self.res_blocks.append(
                MRFModule(
                    hparams, ch,ch
                )
            )

        
        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(
                ch, 1, 7, 1, padding=3
            )
        )

    def forward(self, inputs):
        x = self.conv_pre(inputs)
        # print(x.size())
        for i in range(len(self.ups)):
            x = nn.functional.leaky_relu(x, 0.1)
            x  = self.ups[i](x)
            # print(x.size())
            x = self.res_blocks[i](x)

        
        x  = nn.functional.leaky_relu(x, 0.1)

        x = self.conv_post(x)
        x = torch.tanh(x)
        return x



