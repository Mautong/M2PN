import torch
import torch.nn as nn
from torch.autograd import Variable
from layer import FCALayer
import torch.nn.functional as F
import shiftnet
import argparse
# from ptflops import get_model_complexity_info
# from thop import profile
# from torchstat import stat
# from fvcore.nn import FlopCountAnalysis, parameter_count_table


# ----------------------------------------
#         SIMAM
# ----------------------------------------
class SimamAttention(torch.nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SimamAttention, self).__init__()
        self.activation = nn.Sigmoid()
        self.epsilon = epsilon

    def forward(self, x):

        batch_size, channels, height, width = x.size()
        n = width * height - 1
        
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        attention = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.epsilon)) + 0.5
        
        return x * self.activation(attention)

# ----------------------------------------
#         M2PN
# ----------------------------------------
class M2PN(nn.Module):
 
    
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(M2PN, self).__init__()
        self.num_pyramid_levels = recurrent_iter
        self.use_GPU = use_GPU

        self.input_projection = nn.Sequential(
            shiftnet.ShiftConv2d(6, 32),
            nn.ReLU()
        )

        self.update_gates = nn.ModuleDict({
            'input_gate': nn.Sequential(
                nn.Conv2d(32 + 32, 32, 3, 1, 1),
                nn.Sigmoid()
            ),
            'forget_gate': nn.Sequential(
                nn.Conv2d(32 + 32, 32, 3, 1, 1),
                nn.Sigmoid()
            ),
            'cell_gate': nn.Sequential(
                nn.Conv2d(32 + 32, 32, 3, 1, 1),
                nn.Tanh()
            ),
            'output_gate': nn.Sequential(
                nn.Conv2d(32 + 32, 32, 3, 1, 1),
                nn.Sigmoid()
            )
        })

        self.fca = FCALayer(channel=32, dct_h=14, dct_w=14)

        self.refinement_modules = nn.ModuleList([
            self._make_refinement_module(32, 32) for _ in range(5)
        ])

        self.spatial_attention = SimamAttention(epsilon=1e-4)

        self.image_reconstruction = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def _make_refinement_module(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )

    def _init_pyramid_states(self, batch_size, height, width):

        h_state = Variable(torch.zeros(batch_size, 32, height, width))
        c_state = Variable(torch.zeros(batch_size, 32, height, width))
        
        if self.use_GPU:
            h_state = h_state.cuda()
            c_state = c_state.cuda()
            
        return h_state, c_state

    def forward(self, input_tensor):

        batch_size, _, height, width = input_tensor.size()
        

        current_estimate = input_tensor
        h_state, c_state = self._init_pyramid_states(batch_size, height, width)
        

        pyramid_outputs = []


        for _ in range(self.num_pyramid_levels):

            x = torch.cat((input_tensor, current_estimate), 1)
            x = self.input_projection(x)


            combined_state = torch.cat((x, h_state), 1)
            

            i_gate = self.update_gates['input_gate'](combined_state)
            f_gate = self.update_gates['forget_gate'](combined_state)
            g_gate = self.update_gates['cell_gate'](combined_state)
            o_gate = self.update_gates['output_gate'](combined_state)
            

            c_state = f_gate * c_state + i_gate * g_gate
            h_state = o_gate * torch.tanh(c_state)


            x = h_state
            

            for module in self.refinement_modules:
                residual = x
                x = F.relu(self.fca(module(x)) + residual)
            

            x = self.spatial_attention(x)
            

            current_estimate = self.image_reconstruction(x)
            pyramid_outputs.append(current_estimate)


        return current_estimate, pyramid_outputs

# 计算FLOPS

# model=M2PN(recurrent_iter=6, use_GPU=False)


# #ptflops
# macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,print_per_layer_stat=True, verbose=True)
# print('{:<30} {:<8}'.format('Computational complexity:', macs))
# print('{:<30} {:<8}'.format('Number of parameters: ', params))

# thop
# dummy_input=torch.randn(1, 3, 512, 512)
# macs, params = profile(model, (dummy_input,), ret_layer_info=False)
# print('flops: ', macs, 'params: ', params)
# print('flops: %.2f G, params: %.2f K' % (macs / 1000000000.0, params / 1000.0))

#torchstat
# stat(model, (3, 512, 512))

#fvcore
# tensor = (torch.rand(1, 3, 512, 512),)
# flops = FlopCountAnalysis(model, tensor)
# print("FLOPs: ", flops.total())
# print(parameter_count_table(model))
