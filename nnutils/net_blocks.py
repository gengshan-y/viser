import torch
import torchvision
import torch.nn as nn
import sys

class CodePredictorTex(nn.Module):
    def __init__(self, nz_feat=100,tex_code_dim=64, shape_code_dim=64):
        super(CodePredictorTex, self).__init__()
        self.tex_predictor = nn.Linear(nz_feat, tex_code_dim)
        self.shape_predictor = nn.Linear(nz_feat, shape_code_dim)

    def forward(self, feat):
        tex_code = self.tex_predictor.forward(feat)
        shape_code = self.shape_predictor.forward(feat)
        return tex_code, shape_code

