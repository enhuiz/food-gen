import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from linear_attention_transformer import ImageLinearAttention

EPS = 1e-8


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


attn_and_ff = lambda chan: nn.Sequential(
    *[
        Residual(Rezero(ImageLinearAttention(chan))),
        Residual(
            Rezero(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1),
                    leaky_relu(),
                    nn.Conv2d(chan * 2, chan, 1),
                )
            )
        ),
    ]
)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if type(x) is list:
            x = [self(xk).unsqueeze(1).expand(-1, lk, -1) for xk, lk in x]
            return torch.cat(x, dim=1)

        x = F.normalize(x, dim=-1)

        return self.net(x)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3
        self.conv = ModulatedConv2d(input_channel, out_filters, 1, demodulate=False)

        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsample
            else None
        )

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        demodulate=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.demodulate = demodulate

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channel))

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
        )
        bias = self.bias.expand(batch, -1).flatten(0)
        out = F.conv_transpose2d(
            input, weight, bias, padding=self.padding, stride=1, groups=batch
        )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_channels,
        filters,
        upsample=True,
        upsample_rgb=True,
        simple=True,
    ):
        super().__init__()
        self.simple = simple

        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsample
            else nn.Identity()
        )

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Conv2d(1, filters, 1, bias=False)
        self.conv1 = ModulatedConv2d(input_channels, filters, 3)

        if not simple:
            self.to_style2 = nn.Linear(latent_dim, filters)
            self.to_noise2 = nn.Conv2d(1, filters, 1, bias=False)
            self.conv2 = ModulatedConv2d(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.to_noise1.weight)
        try:
            nn.init.zeros_(self.to_noise2.weight)
        except:
            pass

    def forward(self, x, prev_rgb, istyle, inoise):
        x = self.upsample(x)
        inoise = inoise[..., : x.shape[-2], : x.shape[-1]]

        noise1 = self.to_noise1(inoise)
        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        if not self.simple:
            noise2 = self.to_noise2(inoise)
            style2 = self.to_style2(istyle)
            x = self.conv2(x, style2)
            x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)

        return x, rgb


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample=True,
        simple=True,
    ):
        super().__init__()
        self.simple = simple
        self.conv_res = nn.Conv2d(
            input_channels, filters, 1, stride=(2 if downsample else 1)
        )

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            *(
                tuple()
                if simple
                else (nn.Conv2d(filters, filters, 3, padding=1), leaky_relu())
            ),
        )

        self.downsample = (
            nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class Generator(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim,
        style_depth=8,
        capacity=16,
        attn_layers=[],
        fmap_max=512,
        lr_mul=0.01,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(math.log2(image_size) - 1)

        filters = [capacity * (2 ** (i + 1)) for i in range(self.num_layers)]
        filters = filters[::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.style_vectorizer = StyleVectorizer(latent_dim, style_depth, lr_mul)

        self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        styles = self.style_vectorizer(styles)

        batch_size = styles.shape[0]
        image_size = self.image_size

        x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb


class Discriminator(nn.Module):
    def __init__(
        self,
        image_size,
        capacity=16,
        attn_layers=[],
        fmap_max=512,
    ):
        super().__init__()
        num_layers = int(math.log2(image_size) - 1)
        num_init_filters = 3

        blocks = []
        filters = [num_init_filters] + [(64) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        for block, attn_block in zip(self.blocks, self.attn_blocks):
            x = block(x)

            if attn_block is not None:
                x = attn_block(x)

        x = self.final_conv(x)
        x = x.flatten(1)
        x = self.to_logit(x)

        return x.squeeze()


if __name__ == "__main__":
    from torchsummary import summary

    G = Generator(128, 64)
    summary(G, [(6, 64), (128, 128, 1)], device="cpu")
    summary(Discriminator(128), (3, 128, 128), device="cpu")
