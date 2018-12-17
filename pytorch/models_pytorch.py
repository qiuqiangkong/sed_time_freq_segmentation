import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """
    
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    
    
def tile(a, dim, n_tile, cuda=True):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    if cuda:
        order_index = order_index.cuda()
    return torch.index_select(a, dim, order_index)
    
    
class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(VggishConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input):
        
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x
    
        
class VggishBottleneck(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        
        super(VggishBottleneck, self).__init__()
        
        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=32)
        self.conv_block2 = VggishConvBlock(in_channels=32, out_channels=64)
        self.conv_block3 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block4 = VggishConvBlock(in_channels=128, out_channels=128)

        self.final_conv = nn.Conv2d(in_channels=128, out_channels=classes_num,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), bias=True)

        self.init_weights()
        
    def init_weights(self):

        init_layer(self.final_conv)

    def forward(self, input):
        
        (_, seq_len, freq_bins) = input.shape

        x = input.view(-1, 1, seq_len, freq_bins)
        '''(samples_num, feature_maps, time_steps, freq_bins)'''

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        bottleneck = F.sigmoid(self.final_conv(x))
        '''(samples_num, classes_num, time_steps, freq_bins)'''
        
        return bottleneck
        
        
class VggishGMP(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        
        super(VggishGMP, self).__init__()
        
        self.bottleneck = VggishBottleneck(classes_num, seq_len, freq_bins, cuda)
        
    def forward(self, input, return_bottleneck=False):
        """Forward function. 
        
        Args:
          input: (batch_size, time_steps, freq_bins)
          return_bottleneck: bool
          
        Returns:
          output: (batch_size, classes_num)
          bottleneck (optional): (batch_size, classes_num, time_steps, freq_bins)
        """
        
        bottleneck = self.bottleneck(input)
        '''(batch_size, classes_num, time_steps, freq_bins)'''
        
        # Pool each feature map to a scalar. 
        output = self.global_max_pooling(bottleneck)
        '''(batch_size, classes_num)'''
        
        if return_bottleneck:
            return output, bottleneck
            
        else:
            return output
            
    def global_max_pooling(self, input):
        
        x = F.max_pool2d(input, kernel_size=input.shape[2:])
        output = x.view(x.shape[0], x.shape[1])
        
        return output
        
        
class VggishGAP(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        
        super(VggishGAP, self).__init__()
        
        self.bottleneck = VggishBottleneck(classes_num, seq_len, freq_bins, cuda)
        
    def forward(self, input, return_bottleneck=False):
        """Forward function. 
        
        Args:
          input: (batch_size, time_steps, freq_bins)
          return_bottleneck: bool
          
        Returns:
          output: (batch_size, classes_num)
          bottleneck (optional): (batch_size, classes_num, time_steps, freq_bins)
        """
        
        bottleneck = self.bottleneck(input)
        '''(batch_size, classes_num, time_steps, freq_bins)'''
        
        # Pool each feature map to a scalar. 
        output = self.global_avg_pooling(bottleneck)
        '''(batch_size, classes_num)'''
        
        if return_bottleneck:
            return output, bottleneck
            
        else:
            return output
            
    def global_avg_pooling(self, input):
        
        x = F.avg_pool2d(input, kernel_size=input.shape[2:])
        output = x.view(x.shape[0], x.shape[1])
        
        return output
        
        
class VggishGWRP(nn.Module):
    def __init__(self, classes_num, seq_len, freq_bins, cuda):
        
        super(VggishGWRP, self).__init__()
        
        self.seq_len = seq_len
        self.freq_bins = freq_bins
        
        self.bottleneck = VggishBottleneck(classes_num, seq_len, freq_bins, cuda)
                               
        decay = 0.9998
        (self.gwrp_w, self.sum_gwrp_w) = self.calculate_gwrp_weights(decay, cuda)
        
    def calculate_gwrp_weights(self, decay, cuda):
        
        gwrp_w = decay ** np.arange(self.seq_len * self.freq_bins)
        gwrp_w = torch.Tensor(gwrp_w)
        
        if cuda:
            gwrp_w = gwrp_w.cuda()
            
        sum_gwrp_w = torch.sum(gwrp_w)
        
        return gwrp_w, sum_gwrp_w
        
    def forward(self, input, return_bottleneck=False):
        """Forward function. 
        
        Args:
          input: (batch_size, time_steps, freq_bins)
          return_bottleneck: bool
          
        Returns:
          output: (batch_size, classes_num)
          bottleneck (optional): (batch_size, classes_num, time_steps, freq_bins)
        """
        
        bottleneck = self.bottleneck(input)
        '''(batch_size, classes_num, time_steps, freq_bins)'''
        
        # Pool each feature map to a scalar. 
        output = self.global_weighted_rank_pooling(bottleneck)
        '''(batch_size, classes_num)'''
        
        if return_bottleneck:
            return output, bottleneck
            
        else:
            return output
        
    def global_weighted_rank_pooling(self, input):
        """Global weighted rank pooling. 
        
        Args:
          input: (batch_size, classes_num, time_steps, freq_bins)
          
        Returns:
          output: (batch_size, classes_num)
        """

        x = input.view((input.shape[0], input.shape[1], 
                        input.shape[2] * input.shape[3]))
        '''(batch_size, classes_num, time_steps * freq_bins)'''
        
        # Sort values in each feature map in descending order. 
        (x, _) = torch.sort(x, dim=-1, descending=True)
        
        x = x * self.gwrp_w[None, None, :]
        '''(batch_size, classes_num, time_steps * freq_bins)'''
        
        x = torch.sum(x, dim=-1)
        '''(batch_size, classes_num)'''
        
        output = x / self.sum_gwrp_w
        
        return output
        
        
def get_model(model_type):
    
    if model_type == 'gmp': 
        return VggishGMP
        
    elif model_type == 'gap': 
        return VggishGAP
        
    elif model_type == 'gwrp':
        return VggishGWRP
        
    else:
        raise Exception('Incorrect model type!')