"""
UniTok: A Unified Tokenizer for Visual Generation and Understanding.
Simplified implementation of VQ-GAN / VQ-VAE for NanoChat.

Architecture:
- Encoder: CNN that downsamples images to a grid of latent vectors.
- Quantizer: Vector Quantization to map latents to a codebook.
- Decoder: CNN that upsamples quantized latents back to image.
- Discriminator: PatchGAN for adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(32, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return self.act(h + self.shortcut(x))

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, n_res_layers=2, downsample_factors=[2, 2, 2, 2]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)

        layers = []
        c = hidden_dim
        for ds in downsample_factors:
            for _ in range(n_res_layers):
                layers.append(ResBlock(c))
            layers.append(nn.Conv2d(c, c, 4, stride=ds, padding=1)) # Downsample

        layers.append(ResBlock(c))
        layers.append(nn.GroupNorm(32, c))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(c, 256, 3, padding=1)) # Project to latent dim (standard VQGAN uses 256)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv_in(x)
        return self.net(h)

class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=128, n_res_layers=2, upsample_factors=[2, 2, 2, 2]):
        super().__init__()
        c = 256 # Latent dim
        self.conv_in = nn.Conv2d(c, hidden_dim, 3, padding=1)

        layers = []
        c = hidden_dim
        # Reverse order for upsampling
        for us in reversed(upsample_factors):
            layers.append(ResBlock(c))
            layers.append(nn.Upsample(scale_factor=us, mode='nearest'))
            layers.append(nn.Conv2d(c, c, 3, padding=1))

        for _ in range(n_res_layers):
            layers.append(ResBlock(c))

        layers.append(nn.GroupNorm(32, c))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(c, out_channels, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.net(x)
        return torch.tanh(x) # Images in [-1, 1]

class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantizer (Euclidean).
    """
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e # codebook size
        self.e_dim = e_dim # embedding dimension
        self.beta = beta # commitment cost

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # z: (B, C, H, W) -> (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2ze
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z)**2) + \
               torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # (B, H, W, C) -> (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        # indices: (B*H*W) or (B, H, W)
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    """
    def __init__(self, in_channels=3, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = []
        # No norm in first layer
        layers.append(nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        c = hidden_dim
        for i in range(1, n_layers):
            stride = 2
            layers.append(nn.Conv2d(c, c*2, 4, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(c*2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            c *= 2

        layers.append(nn.Conv2d(c, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class UniTok(nn.Module):
    def __init__(self, vocab_size=16384, embed_dim=256, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim)
        self.quantizer = VectorQuantizer(vocab_size, embed_dim)
        self.quant_conv = nn.Conv2d(256, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, 256, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_codes(self, codes, shape):
        # codes: (B, H, W)
        quant = self.quantizer.get_codebook_entry(codes, None) # returns (B, H, W, C)
        quant = quant.permute(0, 3, 1, 2) # (B, C, H, W)
        dec = self.decode(quant)
        return dec

    def forward(self, x):
        quant, diff, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

class VQLoss(nn.Module):
    def __init__(self, disc_start=1000, codebook_weight=1.0, pixelloss_weight=1.0, disc_weight=0.8, perceptual_weight=1.0):
        super().__init__()
        self.disc_start = disc_start
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight

        self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
        # Freezing LPIPS
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            return 1.0

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, split="train", discriminator=None, logits_fake=None, logits_real=None):
        # inputs: [-1, 1]
        rec_loss = torch.abs(inputs - reconstructions)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = torch.mean(rec_loss)

        # Update Generator
        if optimizer_idx == 0:
            # GAN Part
            if discriminator is not None and global_step > self.disc_start:
                # Generator wants logits_fake to be 1 (real)
                # But here logits_fake is the D output on reconstruction
                # If we pass logits_fake here, it's computed outside.
                # Actually, usually we compute D(rec) here.
                logits_fake = discriminator(reconstructions)
                g_loss = -torch.mean(logits_fake)

                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0)
            else:
                g_loss = torch.tensor(0.0, device=inputs.device)
                d_weight = torch.tensor(0.0, device=inputs.device)

            loss = nll_loss + self.codebook_weight * codebook_loss + self.disc_weight * d_weight * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach()}
            return loss, log

        # Update Discriminator
        if optimizer_idx == 1:
            if global_step > self.disc_start:
                 # Real inputs -> 1, Fake inputs -> 0
                 # logits_real = discriminator(inputs.detach())
                 # logits_fake = discriminator(reconstructions.detach())

                 # Hinge loss
                 loss_real = torch.mean(F.relu(1. - logits_real))
                 loss_fake = torch.mean(F.relu(1. + logits_fake))
                 d_loss = 0.5 * (loss_real + loss_fake)
            else:
                 d_loss = torch.tensor(0.0, device=inputs.device)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}
            return d_loss, log
