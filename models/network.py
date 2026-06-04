import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .cbam import CBAM

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ExplainableMultiTaskNet(nn.Module):
    def __init__(self, num_classes=3, minority_class_idx=2):
        super().__init__()
        # 1. Shared Encoder: EfficientNet-B6
        self.encoder = timm.create_model('tf_efficientnet_b6_ns', features_only=True, pretrained=True)
        enc_chs = [56, 72, 144, 200, 576] # Feature channels for EfficientNet-B6 stages
        
        # 2. CBAM Attention at Bottleneck
        self.cbam = CBAM(in_planes=enc_chs[4])
        
        # Hyperparams for Latent Augmentation (Section 3.1.2)
        self.minority_class_idx = minority_class_idx
        self.alpha = 0.1
        self.sigma = 0.01

        # 3. UNet++ Decoder (Dense Skip Connections)
        dec_chs = [64, 128, 256, 512, 1024]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # UNet++ Nodes (Simplified representation of nested dense blocks)
        self.conv0_1 = ConvBlock(enc_chs[0] + enc_chs[1], dec_chs[0])
        self.conv1_1 = ConvBlock(enc_chs[1] + enc_chs[2], dec_chs[1])
        self.conv2_1 = ConvBlock(enc_chs[2] + enc_chs[3], dec_chs[2])
        self.conv3_1 = ConvBlock(enc_chs[3] + enc_chs[4], dec_chs[3])

        self.conv0_2 = ConvBlock(enc_chs[0] + dec_chs[0]*1 + dec_chs[1], dec_chs[0])
        self.conv1_2 = ConvBlock(enc_chs[1] + dec_chs[1]*1 + dec_chs[2], dec_chs[1])
        self.conv2_2 = ConvBlock(enc_chs[2] + dec_chs[2]*1 + dec_chs[3], dec_chs[2])

        self.conv0_3 = ConvBlock(enc_chs[0] + dec_chs[0]*2 + dec_chs[1], dec_chs[0])
        self.conv1_3 = ConvBlock(enc_chs[1] + dec_chs[1]*2 + dec_chs[2], dec_chs[1])

        self.conv0_4 = ConvBlock(enc_chs[0] + dec_chs[0]*3 + dec_chs[1], dec_chs[0])
        
        # Multi-scale Deep Supervision Projection Heads (1x1 convs)
        self.ds_head1 = nn.Conv2d(dec_chs[3], 1, kernel_size=1) # 1/16
        self.ds_head2 = nn.Conv2d(dec_chs[2], 1, kernel_size=1) # 1/8
        self.ds_head3 = nn.Conv2d(dec_chs[1], 1, kernel_size=1) # 1/4
        self.ds_head4 = nn.Conv2d(dec_chs[0], 1, kernel_size=1) # 1/2
        
        # Final Segmentation Head
        self.seg_head = nn.Conv2d(dec_chs[0], 1, kernel_size=1)

        # 4. Classification Head (Eq 3.11 & 3.12)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(enc_chs[4], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def latent_augmentation(self, z, labels):
        """ Inject stochastic perturbations for minority classes (Eq 3.9) """
        if labels is not None and self.training:
            # Mask for minority class
            mask = (labels == self.minority_class_idx).view(-1, 1, 1, 1).float()
            epsilon = torch.randn_like(z) * self.sigma
            perturbation = self.alpha * epsilon * mask
            return z + perturbation
        return z

    def forward(self, x, labels=None):
        # 1. Encoder Features
        features = self.encoder(x)
        x0_0, x1_0, x2_0, x3_0, bottleneck = features[0], features[1], features[2], features[3], features[4]
        
        # 2. Bottleneck Attention & Latent Augmentation
        z_attn = self.cbam(bottleneck)
        z_aug = self.latent_augmentation(z_attn, labels)
        x4_0 = z_aug

        # 3. UNet++ Decoder
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Multi-scale outputs (before sigmoid, handled in loss)
        ds1 = self.ds_head1(x3_1)
        ds2 = self.ds_head2(x2_2)
        ds3 = self.ds_head3(x1_3)
        ds4 = self.ds_head4(x0_4)
        
        seg_out = self.seg_head(x0_4) # Final segmentation output

        # 4. Classification Output
        cls_feat = self.gap(x4_0).view(x.size(0), -1)
        cls_out = self.classifier(cls_feat)

        if self.training:
            return cls_out, seg_out, [ds1, ds2, ds3, ds4]
        else:
            # During inference, return sigmoid for segmentation masks
            return cls_out, torch.sigmoid(seg_out)