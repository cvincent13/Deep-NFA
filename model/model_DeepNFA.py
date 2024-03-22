import torch
import torch.nn as nn



class NFABlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.significance = Significance()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.significance(out).unsqueeze(1)
        return out
    

class SpatialNFABlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        # to be completed
        #self.sasa = SASABlock(?)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.significance = Significance()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # to be completed
        # out = self.sasa(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.significance(out)
        return out
    

# to be completed
class SASABlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x):
        return x
    

def ln_gamma_approx(a, x):
    return (a-1)*torch.log(x) - x + torch.log(1 + (a-1)/(x+1e-8) + (a-1)*(a-2)/(torch.pow(x,2)+1e-8))

def significance_score(x, n_pixels, n_channels):
    # Covariance matrix with diagonal assumption
    sigma = torch.std(x, dim=(-1,-2), keepdim=True)
    return torch.clamp(torch.lgamma(n_channels/2) -torch.log(n_pixels) - ln_gamma_approx(n_channels/2,(torch.pow(x/torch.sqrt(sigma+1e-8),2).sum(1))/2), min=0.)

class Significance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_pixels = torch.tensor(x.size(-1)*x.size(-2))
        n_channels = torch.tensor(x.size(1))
        return significance_score(x, n_pixels, n_channels)
    

def sigm_alpha(x, alpha, n_pixels):
    return 2/(1+torch.exp(-alpha*(x + torch.log(n_pixels)))) - 1

class SIGMalpha(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        n_pixels = torch.tensor(x.size(-1)*x.size(-2))
        return sigm_alpha(x, self.alpha, n_pixels)
    

class ECABlock(nn.Module):
    def __init__(self, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)    
    

class DNSC(nn.Module):
    def __init__(self, input_channels, block, num_blocks, nb_filter, deep_supervision=False):
        super(DNSC, self).__init__()
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])


    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [x4_0, x3_1, x2_2, x1_3, x0_4, x0_1, x0_2, x0_3]
        else:
            return [x4_0, x3_1, x2_2, x1_3, x0_4]


class DeepNFA(nn.Module):
    def __init__(self, input_channels, block, num_blocks, nb_filter, alpha):
        super(DeepNFA, self).__init__()
        self.backbone = DNSC(input_channels, block, num_blocks, nb_filter)

        self.nfa_block_1 = NFABlock(n_channels=nb_filter[0])
        self.nfa_block_2 = NFABlock(n_channels=nb_filter[1])
        self.nfa_block_3 = NFABlock(n_channels=nb_filter[2])
        self.nfa_block_4 = NFABlock(n_channels=nb_filter[3])
        self.nfa_block_5 = NFABlock(n_channels=nb_filter[4])

        # to be completed
        # spatial blocks

        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.eca_block = ECABlock()
        self.sigm_alpha = SIGMalpha(alpha)

    def forward(self, input, visualization=False):
        x4_0, x3_1, x2_2, x1_3, x0_4 = self.backbone(input)

        sign_scores_1 = self.nfa_block_1(x0_4)
        sign_scores_2 = self.up(self.nfa_block_2(x1_3))
        sign_scores_3 = self.up_4(self.nfa_block_3(x2_2))
        sign_scores_4 = self.up_8(self.nfa_block_4(x3_1))
        sign_scores_5 = self.up_16(self.nfa_block_5(x4_0))

        sign_scores = torch.cat([sign_scores_1, sign_scores_2, sign_scores_3, sign_scores_4, sign_scores_5], dim=1)
        sign_scores_weighted = self.eca_block(sign_scores)

        significance = sign_scores_weighted.max(dim=1)[0]
        significance = self.sigm_alpha(significance)

        if visualization:
            return significance, sign_scores_weighted.detach().cpu().numpy(), sign_scores.detach().cpu().numpy()
        else:
            return significance