"""GAN-based classifier.

  * Generator G   : encoder-decoder (generator-style).  The encoder is the feature extractor
                    (256 -> 8x8x256 bottleneck); the decoder reconstructs the image.
  * Discriminator D: works on encoder features and has TWO heads
                       - adv head : real / fake logit (adversarial training signal)
                       - cls head : 19-way tissue classifier  <- this is what forward() returns

forward(x) -> logits (B, 19) in both train and eval mode.  During training use
`fit_batch`, which optimises CE + reconstruction + hinge-GAN losses with two optimisers
(opt_g for the generator, opt_d for the discriminator).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import NUM_CLASSES, BaseClassifier, init_weights


def down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))


def up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Generator(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        # encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(down(3, base), down(base, base * 2), down(base * 2, base * 4),
                                     down(base * 4, base * 8), down(base * 8, base * 8))
        self.out_ch = base * 8
        self.decoder = nn.Sequential(up(base * 8, base * 8), up(base * 8, base * 4), up(base * 4, base * 2),
                                     up(base * 2, base), nn.ConvTranspose2d(base, 3, 4, 2, 1), nn.Tanh())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f)


class Discriminator(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.adv = nn.Conv2d(256, 1, 3, 1, 1)                                  # PatchGAN real/fake map
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(256 * 2, 256), nn.LeakyReLU(0.2, inplace=True),
                                 nn.Dropout(0.3), nn.Linear(256, num_classes))

    def features(self, f: torch.Tensor) -> torch.Tensor:
        return self.trunk(f)

    def classify(self, f: torch.Tensor) -> torch.Tensor:
        t = self.trunk(f)
        return self.cls(torch.cat([t.mean((2, 3)), t.amax((2, 3))], dim=1))

    def adversarial(self, f: torch.Tensor) -> torch.Tensor:
        return self.adv(self.trunk(f))


class GANClassifier(BaseClassifier):
    key = "gan_classifier"

    def __init__(self, num_classes: int = NUM_CLASSES, base: int = 32,
                 lambda_rec: float = 0.5, lambda_adv: float = 0.05):
        super().__init__()
        self.G = Generator(base)
        self.D = Discriminator(self.G.out_ch, num_classes)
        self.lambda_rec, self.lambda_adv = lambda_rec, lambda_adv
        init_weights(self)

    # ---- shared inference interface ---------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.D.classify(self.G.encode(x))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.G.decode(self.G.encode(x))

    # ---- parameter groups for the two optimisers ---------------------------- #
    def g_parameters(self):
        return self.G.parameters()

    def d_parameters(self):
        return self.D.parameters()

    def fit_batch(self, x: torch.Tensor, y: torch.Tensor, opt_g: torch.optim.Optimizer,
                  opt_d: torch.optim.Optimizer, label_smoothing: float = 0.05):
        """One adversarial training step. Returns (logits, total_loss_value)."""
        self.train()
        opt_g.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)

        feat = self.G.encode(x)
        logits = self.D.classify(feat)
        loss_cls = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        # reconstruction target is the normalised image squashed to the tanh range
        recon = self.G.decode(feat)
        target = torch.tanh(x / 2.5)
        loss_rec = F.l1_loss(recon, target)

        # generator-side adversarial loss: reconstructed images should look "real" to D in feature space
        feat_fake = self.G.encode(recon)
        loss_g_adv = -self.D.adversarial(feat_fake).mean()
        g_total = loss_cls + self.lambda_rec * loss_rec + self.lambda_adv * loss_g_adv
        g_total.backward()

        # discard the adversarial gradients that leaked into D's adv head; recompute properly
        for p in list(self.D.adv.parameters()):
            p.grad = None
        d_real = self.D.adversarial(feat.detach())
        d_fake = self.D.adversarial(feat_fake.detach())
        loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()      # hinge loss
        (self.lambda_adv * loss_d).backward()
        # (trunk grads from the D loss are added on purpose: the trunk is shared by both heads)

        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        opt_g.step()
        opt_d.step()
        return logits.detach(), float(g_total.detach())


def build_model(num_classes: int = NUM_CLASSES) -> GANClassifier:
    return GANClassifier(num_classes)
