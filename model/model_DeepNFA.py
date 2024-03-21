import torch
import torch.nn as nn

from model.model_DNANet import DNSC


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
        out = self.significance(out)
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
    

#torch.special.gammaincc

def ln_gamma_approx(a, x):
    return (a-1)*torch.log(x) - x + torch.log(1 + (a-1)/x + (a-1)*(a-2)/torch.pow(x,2))

def significance_score(x, n_pixels, n_channels):
    return torch.lgamma(n_channels/2) -torch.log(n_pixels) - ln_gamma_approx(n_channels/2,torch.pow(x,2).sum(1)/2)

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


class DeepNFA(nn.Module):
    def __init__(self, input_channels, block, num_blocks, nb_filter, alpha):
        super(DeepNFA, self).__init__()
        self.backbone = DNSC(input_channels, block, num_blocks, nb_filter)

        self.nfa_block_1 = NFABlock(nb_filter[0], nb_filter[0])
        self.nfa_block_2 = NFABlock(nb_filter[0], nb_filter[0])
        self.nfa_block_3 = NFABlock(nb_filter[0], nb_filter[0])
        self.nfa_block_4 = NFABlock(nb_filter[0], nb_filter[0])
        self.nfa_block_5 = NFABlock(nb_filter[0], nb_filter[0])

        # to be completed
        # spatial blocks

        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.eca_block = ECABlock()
        self.sigm_alpha = SIGMalpha(alpha)

    def forward(self, input):
        x4_0, x3_1, x2_2, x1_3, x0_4 = self.backbone(input)

        sign_scores_1 = self.nfa_block_1(x0_4)
        sign_scores_2 = self.up(self.nfa_block_2(x1_3))
        sign_scores_3 = self.up_4(self.nfa_block_3(x2_2))
        sign_scores_4 = self.up_8(self.nfa_block_4(x3_1))
        sign_scores_5 = self.up_16(self.nfa_block_5(x4_0))

        sign_scores = torch.cat([sign_scores_1, sign_scores_2, sign_scores_3, sign_scores_4, sign_scores_5], dim=1)
        sign_scores = self.eca_block(sign_scores)

        significance = sign_scores.min(dim=1)
        significance = self.sigm_alpha(significance)

        return significance