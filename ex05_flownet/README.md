# FlowNetC/FlowNetS Training and Evaluation

This repository contains code for training and evaluating FlowNetC/FlowNetS models.

## Usage

### Training

#### Pre-training

##### FlowNetC

Pre-training on the FlyingThings dataset:

```bash
python train.py --output /your/output/directory --model FlowNetC
```

Checkpoints and tensorboard logs will be written to the specified output directory.
Self-supervised training is possible with the flag
--photometric (and optionally the flag --smoothness_loss).

Pre-training on the FlyingChairs dataset:

```bash
python train.py --output /your/output/directory --model FlowNetC --dataset FlyingChairs
```

Pre-training on FlyingChairs+FlyingThings3D:

```bash
python train.py --output /your/output/directory --model FlowNetC --dataset FlyingChairs --iterations 300000
python train.py --output /your/output/directory --model FlowNetC --dataset FlyingThings3D --restore /path/to/chkpt/checkpoint-train-iter-000300000.pt --completed_iterations 300000 --iterations 600000
```

##### FlowNetS

Pre-training on the FlyingThings dataset:

```bash
python train.py --output /your/output/directory --model FlowNetS
```

Pre-training on FlyingChairs and on FlyingChairs+FlyingThings3D works as described above.

#### Fine-tuning

Fine-tuning is currently supported on the Sintel dataset.

##### FlowNetC

```bash
python train.py --output /your/output/directory --model FlowNetC --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000
```

This would fine-tune the model for 100k iterations in supervised mode.

For self-supervised fine-tuning with a photometric loss, run:

```bash
python train.py --output /your/output/directory --model FlowNetC --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000 --photometric
```

To include the smoothness loss, run:

```bash
python train.py --output /your/output/directory --model FlowNetC --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000 --photometric --smoothness_loss
```

##### FlowNetS

```bash
python train.py --output /your/output/directory --model FlowNetS --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000
```

For self-supervised fine-tuning with a photometric loss, run:

```bash
python train.py --output /your/output/directory --model FlowNetS --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000 --photometric
```

To include the smoothness loss, run:

```bash
python train.py --output /your/output/directory --model FlowNetS --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000 --photometric --smoothness_loss
```

### Evaluation

#### Evaluate FlowNetC on FlyingThings

```bash
python eval.py --output /your/output/directory --model FlowNetC --dataset FlyingThings3D --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt
```

Evaluation results will be written to the specified output directory.
Qualitative results are written to Tensorboard.

#### Evaluate FlowNetS on FlyingThings

```bash
python eval.py --output /your/output/directory --model FlowNetS --dataset FlyingThings3D --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt
```

#### Evaluate FlowNetC on Sintel

```bash
python eval.py --output /your/output/directory --model FlowNetC --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt
```

#### Evaluate FlowNetS on Sintel

```bash
python eval.py --output /your/output/directory --model FlowNetS --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt
```

### Results

Finetuned models are provided under
`/<project_directory>/shared-data/OpticalFlowFinetunedModels/`

All models were pretrained on FlyingChairs for 600k iterations. Results are given as AEPE for the
dataset in the column title.

| Model    | Finetuning                             | Sintel (our test split) | SintelFull | FlyingThings3D |
|----------|----------------------------------------|-------------------------|------------|----------------|
| FlowNetC | no                                     | 0.720                   | 4.361      | 33.136         |
| FlowNetS | no                                     | 0.599                   | 5.061      | 37.280         |
| FlowNetS | Sintel 10k supervised                  | 0.567                   | 3.268      | 37.224         |
| FlowNetS | Sintel 10k photometric+smoothness loss | 0.592                   | 5.130      | 39.012         |

python eval.py --output your_output_dir1 --model FlowNetC --dataset FlyingThings3D --auto_restore pt/flownetc

Note that the photometric loss does not really improve the results.
