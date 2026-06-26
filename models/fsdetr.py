import torch
import torch.nn as nn

from modules.conv import Conv
from modules.block import C2f, RepC3
from modules.custom_block import SPDConv, SNI, CFSB, SHAB
from modules.custom_transformer import DA_AIFI
from modules.head import RTDETRDecoder


class FSDETRModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.b0 = Conv(3, 64, 3, 2)
        self.b1 = Conv(64, 128, 3, 2)
        self.b2 = C2f(128, 128, n=1)

        self.b3 = Conv(128, 256, 3, 2)
        self.b4 = C2f(256, 256, n=1)

        self.b5 = Conv(256, 384, 3, 2)
        self.b6 = SHAB(384, 384, n=1)

        self.b7 = Conv(384, 384, 3, 2)
        self.b8 = SHAB(384, 384, n=3)

        self.h9 = Conv(384, 256, 1, 1, act=False)
        self.h10 = DA_AIFI(256, cm=1024, num_heads=8)
        self.h11 = Conv(256, 256, 1, 1)

        self.h12 = SNI(2)
        self.h13 = Conv(384, 256, 1, 1, act=False)
        self.h15 = RepC3(512, 256, n=3, e=0.5)
        self.h16 = Conv(256, 128, 1, 1)

        self.h17 = SNI(2)
        self.h18 = SPDConv(128, 128)
        self.h20 = CFSB(512, 256, n=1)
        self.h21 = RepC3(256, 256, n=3, e=0.5)

        self.h22 = Conv(256, 256, 3, 2)
        self.h24 = RepC3(384, 256, n=3, e=0.5)

        self.h25 = Conv(256, 256, 3, 2)
        self.h27 = RepC3(512, 256, n=3, e=0.5)

        self.decoder = RTDETRDecoder(
            nc=num_classes,
            ch=(256, 256, 256),
            hd=256,
            nq=300,
            ndp=4,
            nh=8,
            ndl=3,
        )

    def _forward_features(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)

        x3 = self.b3(x2)
        x4 = self.b4(x3)

        x5 = self.b5(x4)
        x6 = self.b6(x5)

        x7 = self.b7(x6)
        x8 = self.b8(x7)

        y9 = self.h9(x8)
        y10 = self.h10(y9)
        y11 = self.h11(y10)

        y12 = self.h12(y11)
        y13 = self.h13(x6)
        y14 = torch.cat([y12, y13], dim=1)
        y15 = self.h15(y14)
        y16 = self.h16(y15)

        y17 = self.h17(y16)
        y18 = self.h18(x2)
        y19 = torch.cat([y18, y17, x4], dim=1)
        y20 = self.h20(y19)
        y21 = self.h21(y20)

        y22 = self.h22(y21)
        y23 = torch.cat([y22, y16], dim=1)
        y24 = self.h24(y23)

        y25 = self.h25(y24)
        y26 = torch.cat([y25, y11], dim=1)
        y27 = self.h27(y26)

        return [y21, y24, y27]

    def forward(self, x, batch=None, return_raw=False):
        feats = self._forward_features(x)

        if return_raw:
            old_state = self.decoder.training
            self.decoder.train(True)
            out = self.decoder(feats, batch=batch)
            self.decoder.train(old_state)
            return out

        return self.decoder(feats, batch=batch)