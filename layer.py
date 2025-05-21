import math
import torch
import torch.nn as nn

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32','box16',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]  #索引是用来表示需要保留的频率分量的位置的
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]  #根据频域特征的能量分布来选择的，也就是选择能量最大的一些频率分量，因为它们包含了更多的图像信息
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    elif 'box' in method:
        all_bot_indices_x = [0,1,2,3,2,3,4,5,4,3,4,5,5,4,5,5]
        all_bot_indices_y = [0,0,0,0,1,1,0,0,1,2,2,1,2,3,3,4]#0, 1, 5, 6, 7, 13, 14, 15, 16, 17, 25, 26, 29, 30, 37, 40
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)   #判断获取的频率列表长度是否相等
       # assert channel % len(mapper_x) == 0     #确保通道数与数据映射列表长度关系是否合理

        self.num_freq = len(mapper_x)#频率分量的数量就是索引个数，这里是与设置的dcth和dctw与索引相乘之后获得的索引

        # fixed DCT init
        #对应于公式中的H,W,i,j,C
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight   #此时的x是经过pooled之后的，将输入patchsize100适应缩放成了设置的dcth/w，此处与dct权重相点乘，保留低频去除高频

        result = torch.sum(x, dim=[2,3])#将频域特征在H和W上求和，将每个通道的频域特征中的所有频率分量的值相加，得到一个标量，表示该通道的重要程度
        return result

    def build_filter(self, pos, freq, POS):      #对应公式，pos为i或j；freq为对应上设置dct_h/w索引中的x或y；POS为设置的dct_h或dct_w
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) #用到二维DCT变换公式，u_x为公式的h；v_y为公式的w，值由82行返回
        if freq == 0:           #当u_x/v_y遍历到索引里面的0项，freq就为0
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)  #tilesizex是dcth，tilesizey是dctw

        c_part = channel // len(mapper_x)#按理来说是论文里面的C’

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):   #0~dcth
                for t_y in range(tile_size_y):  #0~dctw
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        #二维DCT变换公式，result*result
        return dct_filter

class FCALayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(FCALayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)  #等效于上面的num_freq
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]#根据论文中得到的排名前16的索引，将设置的dcth和w与索引联系，整除7获得一个比例系数
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]#比例系数与将放在暂存变量里面的索引相乘，从而使设置的dcth和w与索引对应上
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)