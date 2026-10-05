# Results

## Compare corrupted versus adapted

### without adaptation

Note some of these results were not seeded so they might be slightly different.

```
corruption strength     1     2     3
defocus_blur         55.8  47.1  31.0
glass_blur           56.2  43.2  20.6
motion_blur          61.4  48.0  29.7
zoom_blur            48.4  38.2  30.7
snow                 53.8  27.3  31.8
frost                59.4  40.6  27.2
fog                  57.8  49.0  37.8
brightness           73.6  71.8  68.3
```

### with running mean and variance adaption

`num_bn_updates = 10`

```
corruption strength     1     2     3
defocus_blur         55.7  45.9  27.3
glass_blur           59.6  50.0  30.8
motion_blur          63.6  53.2  38.9
zoom_blur            55.4  47.2  41.0
snow                 59.2  43.9  44.6
frost                61.8  48.4  37.8
fog                  62.3  57.9  51.0
brightness           73.5  72.4  69.9
```

### Takehome:

For many corruptions the adaptation of the mean and variance helps. Some corruptions do not change
the batch statistics. For them the performance does not change with adaptation. 

## Compare batch sizes

`num_bn_updates = 50`,
only run it for default glass_blur severity 1

- bs = 1  Validation complete. Loss 3.114642 accuracy 41.85%
- bs = 4  Validation complete. Loss 2.039543 accuracy 56.03%
- bs = 16 Validation complete. Loss 1.934974 accuracy 57.61%
- bs = 64 Validation complete. Loss 1.933209 accuracy 57.75%
- No update: 56.2%

### Takehome:

Since the batchnorm statistics - mean and variance - are computed over the samples in the batch, 
the larger the batch size the more stable and meaningful they become.

## Num bn updates experiment

`batch_size = 64`,
only run it for default glass_blur severity 1

```bash
python run_resnet.py --apply_bn --num_bn_updates x --batch_size 16
```

- num_bn_updates 10 Validation complete. Loss 1.772994 accuracy 59.04%
- num_bn_updates 50 Validation complete. Loss 2.039543 accuracy 56.03%
- num_bn_updates 100 Validation complete. Loss 1.926813 accuracy 57.56%

### Takehome

We perform the update on samples of the validation set. The more updates we perform,
the more samples we see. Surprisingly, few updates already give a good performance 
and more do not seem to help. This should be checked for other corruption types.
