# Auto-extracted from notebooks/analysis/ThermalGrader.ipynb cell 1
# (the ORIGINAL big-arch Generator used for the paper's old models, e.g. G_832)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        # you can try use activation function or not
        # return x + self.block(x)
        # x = F.leaky_relu(x + self.block(x), negative_slope=0.2, inplace=False)
        return F.leaky_relu(x + self.block(x), negative_slope=0.2, inplace=False)
        # return x

class Generator(nn.Module):
    """
    Wider & deeper generator.
    - label1, label2 -> 512-D each (deeper MLPs)
    - noise -> 512-D
    - concat -> 1536 -> FC -> (B,1024,8,8)
    - Residual stack at 1024, then 4 upsample stages (to 128) with deep residual stacks
    - final 64->32 conv + residuals -> 1ch + tanh
    """
    def __init__(
        self,
        noise_dim,
        emb_dim,            # kept for backward-compat; not used
        img_shape,
        label_ch=512,       # was 256
        noise_ch=512,       # was 256
        base_ch=1024,       # was 512
        n_res_base=3,       # extra depth at 8x8
        n_res_per=4         # extra depth per upsample stage
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_shape = img_shape
        self.size = img_shape[1] // 16  # 128//16 = 8
        assert self.size == 8, "This definition assumes 128x128 output."

        # ----- label projections: (B,1) -> (B,512), deeper than before
        self.label1_proj = nn.Sequential(
            nn.Linear(1, label_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(label_ch, label_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(label_ch, label_ch), nn.LeakyReLU(0.2, inplace=True),
        )
        self.label2_proj = nn.Sequential(
            nn.Linear(1, label_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(label_ch, label_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(label_ch, label_ch), nn.LeakyReLU(0.2, inplace=True),
        )

        # ----- noise to 512-D (or identity if already)
        self.noise_proj = (nn.Identity() if noise_dim == noise_ch else
            nn.Sequential(
                nn.Linear(noise_dim, noise_ch), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(noise_ch,  noise_ch), nn.LeakyReLU(0.2, inplace=True),
            )
        )

        fused_dim = label_ch * 2 + noise_ch  # 512+512+512 = 1536

        # ----- fuse -> (B, base_ch * 8 * 8)
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, base_ch * self.size * self.size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ----- deep residual stack at 8x8 / base_ch
        self.base_stack = nn.Sequential(*[ResidualBlock(base_ch) for _ in range(n_res_base)])

        # ----- helper to build one upsample stage
        def up_block(cin, cout, n_res):
            layers = [
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),  # x2 size
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            layers += [ResidualBlock(cout) for _ in range(n_res)]
            return nn.Sequential(*layers)

        # 8 -> 16 -> 32 -> 64 -> 128, halving channels each time
        self.up1 = up_block(base_ch,       base_ch // 2, n_res_per)  # 1024 -> 512
        self.up2 = up_block(base_ch // 2,  base_ch // 4, n_res_per)  # 512  -> 256
        self.up3 = up_block(base_ch // 4,  base_ch // 8, n_res_per)  # 256  -> 128
        self.up4 = up_block(base_ch // 8,  base_ch // 16, n_res_per) # 128  -> 64

        # an extra conv + residual stack at 128x128 for more depth before Tanh
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch // 16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(32), ResidualBlock(32),
            nn.Conv2d(32, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, noise, labels):
        labels = labels.float()
        l1 = labels[:, :1]   # (B,1)
        l2 = labels[:, 1:2]  # (B,1)

        h1 = self.label1_proj(l1)       # (B,512)
        h2 = self.label2_proj(l2)       # (B,512)
        z  = self.noise_proj(noise)     # (B,512)

        fused = torch.cat([h1, h2, z], dim=1)          # (B,1536)
        x = self.fc(fused).view(noise.size(0), -1, self.size, self.size)  # (B,1024,8,8)

        x = self.base_stack(x)  # deep at 8x8

        x = self.up1(x)  # -> (B, 512, 16,16)
        x = self.up2(x)  # -> (B, 256, 32,32)
        x = self.up3(x)  # -> (B, 128, 64,64)
        x = self.up4(x)  # -> (B,  64,128,128)

        img = self.tail(x)  # -> (B,1,128,128) in [-1,1]
        return img

