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

![](https://api.wandb.ai/files/enhuiz/food-gen/gc7b1ldp/media/images/fake_11001_6fb1ec2bf61792ea4d26.png)

### VAE

```
tzq config/vanilla/vae.yml train
```

![](https://api.wandb.ai/files/enhuiz/food-gen/wu0fv2ea/media/images/fake_11001_c8e66f957db742ae47b3.png)

### VAE with perceptual loss

```
tzq config/vanilla/vae-perceptual.yml train
```

![](https://api.wandb.ai/files/enhuiz/food-gen/w1jilxp4/media/images/fake_11001_3151b19e172b0e9fe66b.png)

## More Results

Detailed running results can be found [here](https://wandb.ai/enhuiz/food-gen).

## Credits

- The food images are collected from [Weibo](https://weibo.com/3973876838/HwLFEyHv0).
