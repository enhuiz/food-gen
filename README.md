# FoodGen: a collection of generative models for fake foods.

## Setup

```
pip install .
```

## Models (Vanilla Generator)

### GAN

```
tzq config/vanilla/gan.yml train
```

![](image/2020-09-30-16-54-13.png)

### VAE

```
tzq config/vanilla/vae.yml train
```

### VAE with perceptual loss

```
tzq config/vanilla/perceptual-vae.yml train
```

## More Results

Detailed running results can be found [here](https://wandb.ai/enhuiz/food-gen).

## Credits

- The food images are collected from [Weibo](https://weibo.com/3973876838/HwLFEyHv0).
