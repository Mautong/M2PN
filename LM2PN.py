import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import FCALayer
from torch.autograd import Variable
# from ptflops import get_model_complexity_info
# from thop import profile
# from torchstat import stat
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

class LM2PN(nn.Module):
    
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LM2PN, self).__init__()
        self.num_refinement_stages = recurrent_iter
        self.use_GPU = use_GPU

        self.input_adapter = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.feature_pathways = nn.ModuleDict({
            'path1': nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 16, 1)
            ),
            'path2': nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 16, 1)
            )
        })

        self.freq_attention = FCALayer(channel=32, dct_h=14, dct_w=14)

        self.fusion_module = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.ReLU()
        )


        self.enhancement_blocks = nn.ModuleList([
            self._make_res_block(32, 32),
            self._make_res_block(32, 32),
            self._make_res_block(32, 32)
        ])

        self.state_gate_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.state_gate_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.state_gate_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.state_gate_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )

    def _apply_feature_enhancement(self, x):
        features = []
        for block in self.enhancement_blocks:
            residual = x
            x = F.relu(self.freq_attention(block(x)) + residual)
            features.append(x)
        return x, features

    def _extract_multi_path_features(self, x):

        feat1 = self.feature_pathways['path1'](x)
        feat2 = self.feature_pathways['path2'](x)
        combined = torch.cat([feat1, feat2], dim=1)
        return self.fusion_module(combined)

    def _init_states(self, batch_size, height, width):

        memory = Variable(torch.zeros(batch_size, 32, height, width))
        context = Variable(torch.zeros(batch_size, 32, height, width))
        
        if self.use_GPU:
            memory = memory.cuda()
            context = context.cuda()
            
        return memory, context

    def forward(self, input_img):

        batch_size, _, height, width = input_img.size()
        

        current_img = input_img
        memory, context = self._init_states(batch_size, height, width)
        
        all_stage_outputs = []
        all_features = []


        for _ in range(self.num_refinement_stages):

            x = torch.cat((input_img, current_img), 1)
            x = self.input_adapter(x)


            combined = torch.cat((x, memory), 1)
            
            i = self.state_gate_i(combined)
            f = self.state_gate_f(combined)
            g = self.state_gate_g(combined)
            o = self.state_gate_o(combined)
            
            context = f * context + i * g
            memory = o * torch.tanh(context)


            x = memory
            x, stage_features = self._apply_feature_enhancement(x)
            

            enhanced_features = self._extract_multi_path_features(x)
            all_features.append(enhanced_features)


            current_img = self.reconstruction(x)
            all_stage_outputs.append(current_img)


        return all_stage_outputs[-1], all_features


# 计算FLOPS

model=LM2PN(recurrent_iter=6, use_GPU=False)
#
#
# # #ptflops
# macs, params = get_model_complexity_info(model, (3, 98, 98), as_strings=True,print_per_layer_stat=True, verbose=True)
# print('{:<30} {:<8}'.format('Computational complexity:', macs))
# print('{:<30} {:<8}'.format('Number of parameters: ', params))

#thop
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