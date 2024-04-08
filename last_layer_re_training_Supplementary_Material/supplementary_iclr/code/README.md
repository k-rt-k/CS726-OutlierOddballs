# Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations

Here we provide the code used to produce the results in our submission.

## File Structure

```
.
+-- notebooks/
|   +-- data_generation/
|   |   +-- generate_imagenet9_bg_paintings.ipynb (ImageNet-9 Paintings-BG data)
|   |   +-- generate_waterbirds_fg_bg.ipynb (Generate FG and BG-Only Waterbirds data)
|   +-- imagenet_stylized/
|   |   +-- imagenet_stylized_dfr_train.ipynb (Train DFR on Stylized Imagenet)
|   |   +-- imagenet_stylized_dfr_evaluate.ipynb (Evaluate DFR on Stylized Imagenet)
|   +-- imagennet9_dfr.ipynb (Train and evaluate DFR on ImageNet-9 BG challenge)
|   +-- in_to_in9.json (ImageNet to ImageNet-9 class maping)
+-- celeba_metadata.csv (Metadata file for CelebA)
+-- train_classifier.py (Train base models on CelebA and WaterBirds)
+-- utils.py (Utility functions)
+-- wb_data.py (CelebA and WaterBirds dataloaders)
+-- imagenet_datasets.py (Dataloaders for ImageNet variations)
+-- imagenet_extract_embeddings.py (Extract embeddings from an ImageNet-like dataset)
+-- dfr_evaluate_spurious.py (Tune and evaluate DFR for a given base model)
```

## Data access

- Waterbirds: see instructions [here](https://github.com/kohpangwei/group_DRO#waterbirds).
- CelebA: see instruction [here](https://github.com/kohpangwei/group_DRO#celeba).
- ImageNet and Stylized ImageNet: see instruction [here](https://github.com/rgeirhos/Stylized-ImageNet#usage).
- ImageNet-C: see instruction [here](https://github.com/hendrycks/robustness).
- ImageNet-R: see instruction [here](https://github.com/hendrycks/imagenet-r).
- Background Challenge: see instruction [here](https://github.com/MadryLab/backgrounds_challenge).

For CelebA, please copy the `celeba_metadata.csv` from this repo to the root
folder containing the CelebA dataset and rename it to `metadata.csv`.

We provide jupyter notebooks to generate the Paintings-BG split of ImageNet-9
and aligned FG-Only, BG-Only and Original Waterbirds splits in 
`notebooks/data_generation/`.

## Example comands: spurious correlation benchmarks 

### Base models

To train base models on CelebA and Waterbirds, use the following commands.
```bash
# Waterbirds
python3 train_classifier.py --output_dir=<OUTPUT_DIR> --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=1 --data_dir=<WATERBIRDS_DIR> --test_wb_dir=<WATERBIRDS_DIR> \
  --augment_data --seed=<SEED>

# CelebA
python3 train_classifier.py --output_dir=<OUTPUT_DIR> --pretrained_model \
  --num_epochs=50 --weight_decay=1e-4 --batch_size=128 --init_lr=1e-3 \
  --eval_freq=1 --data_dir=<CELEBA_DIR> --test_wb_dir=<CELEBA_DIR> \
  --augment_data --seed=<SEED>
```

Here `OUTPUT_DIR` is a path to the folder where the logs will be stored,
`WATERBIRDS_DIR` and `CELEBA_DIR` are the directories containing waterbirds
and CelebA data respectively, and `SEED` is the random seed.

To train base models without minority groups (for DFR_{TR-NM}^{TR}), use the
following commands.
```bash
# Waterbirds
python3 train_classifier.py ---output_dir=<OUTPUT_DIR> --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=1 --data_dir=<WATERBIRDS_DIR> --test_wb_dir=<WATERBIRDS_DIR> \
  --augment_data --seed=<SEED> num_minority_groups_remove=2

# CelebA
python3 train_classifier.py --output_dir=<OUTPUT_DIR> --pretrained_model \
  --num_epochs=50 --weight_decay=1e-4 --batch_size=128 --init_lr=1e-3 \
  --eval_freq=1 --data_dir=<CELEBA_DIR> --test_wb_dir=<CELEBA_DIR> \
  --augment_data --seed=<SEED> --num_minority_groups_remove=1
```

You can train models without ImageNet-pretrained initialization by removing
the `--pretrained_model` flag.
You can disable data augmentation by removing the `--augment_data` flag.
You can change the number of epochs, weight decay, learning rate and batch size
with the `--num_epochs`, `--weight_decay`, `--init_lr`, and `--batch_size` flags
respectively.

### DFR

You can run DFR (all variations) on the Waterbirds and CelebA data with the 
following commands.

```bash
python3 dfr_evaluate_spurious.py --data_dir=<DATA_DIR> \
  --result_path=<RESULT_PATH.pkl> --ckpt_path=<CKPT_PATH> \
  --tune_class_weights_dfr_train
```

Here `DATA_DIR` is the directory containing Waterbirds or CelebA data,
`RESULT_PATH` is the path where a pickle dump of the results will be saved,
and `CKPT_PATH` is the checkpoint path.
For DFR_{TR-NM}^{TR} do not use the `--tune_class_weights_dfr_train` flag, if you do not
want to tune the class weights.

The script will output the results to console and save them to `RESULT_PATH`.

## ImageNet experiments

### Extracting embeddings 

To reproduce the ImageNet experiments in the paper, you will need to first compute
the embeddings of the data using the base model. 
We provide a `imagenet_extract_embeddings.py` script for this purpose:

```bash
python3 imagenet_extract_embeddings.py --dataset_dir=<DATA_PATH> \
  --split=[val | train] --model=[resnet50 | vitb16] --batch_size=100
```

Here you can specify paths to the desired ImageNet variation folder in place of
`DATA_PATH`.
You can also specify which dataset variation you are using with the `--dataset` flag
with possible values `[imagenet | imagenet-a | imagenet-r | imagenet-c | bg_challenge]`.

The extracted embeddings will be saved in the `<DATA_PATH>` root folder.

### DFR on Background Challenge

We provide a jupyter notebook to reproduce our results on ImageNet-9 Background challenge
data at `notebooks/imagennet9_dfr.ipynb`.

### DFR Texture Bias

We provide a jupyter notebooks to reproduce our results on texture bias
data at `notebooks/imagenet_stylized/`.
First, run `imagenet_stylized_dfr_train.ipynb` to train the DFR models on
Stylized ImageNet variations.
Then, run `imagenet_stylized_dfr_evaluate.ipynb` to evaluate the trained models
on all ImageNet variations.

## References

The `train_classifier.py`, `utils.py` and `wb_data.py` are aadapted from the 
[kohpangwei/group_DRO repo](https://github.com/kohpangwei/group_DRO).
Dataloaders in `imagenet_datasets.py` are adapted from
[MadryLab/backgrounds_challenge repo](https://github.com/MadryLab/backgrounds_challenge)
and 
[rgeirhos/Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet).
We use VIT-B-16 model pre-trained on ImageNet-21k and fine-tuned on ImageNet from
the [lukemelas/PyTorch-Pretrained-ViT repo](https://github.com/lukemelas/PyTorch-Pretrained-ViT).
To evaluate shape bias of the models we use the [bethgelab/model-vs-human repo](https://github.com/bethgelab/model-vs-human).
