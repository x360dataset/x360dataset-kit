<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="statics/favicon.ico" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">360+x : A Panoptic Multi-modal Scene Understanding Dataset</h3>

[The MIx Group](https://mix.jianbojiao.com/), University of Birmingham

<a href="https://mix.jianbojiao.com/"><img height=50 src="statics/mix_group.png" style="padding-left: 10px; padding-right: 10px"/></a>
<a href="https://www.birmingham.ac.uk/"><img height=40 src="statics/UoB_Crest_Logo_RGB_POS_Landscape.png" style="padding-left: 10px; padding-right: 10px"/></a>
<a href="https://www.baskerville.ac.uk/"><img height=50 src="statics/baskerville.svg" style="padding-left: 10px; padding-right: 10px"></a>

<p align="center">
    <br /> 
    <a href="https://x360dataset.github.io/"><strong>Explore the Project Page »</strong></a>
    <br />
    <br />
    <a href="https://github.com/x360dataset/x360dataset-kit/issues">Report Bug</a>
    ·
    <a href="https://github.com/x360dataset/x360dataset-kit/issues">Request Feature</a>
  </p>
</div>


Welcome to [**360+x**](x360dataset.github.io) dataset development kit repo.

### Roadmap

This Development Toolbox is under construction 🚧.

- [x] Code Release - 09/06/2024
- [x] TAL Checkpoints Release - 04/07/2024
- [x] TAL Annotations Release - 13/08/2024
- [x] Extracted Features Extracted by 360x Pretrained Extractor Release - 13/08/2024

### Table of Contents

<ul>
<li>
  <a href="#dataset-highlights">Dataset Highlights</a>
</li>
<li>
  <a href="#dataset-access">Dataset Access</a>
</li>
<li>
  <a href="#toolkit-structure">Toolkit Structure</a>
</li>
<li>
<a href="#training">Training</a>
</li>
<li>
  <a href="#pretrained-models">Pretrained Models</a>
</li>
<li>
  <a href="#features">Features</a>
</li>
<li><a href="#license">License</a></li>
<li><a href="#cite">Cite</a></li>
<li><a href="#contact">Contact</a></li>
<li><a href="#acknowledgments">Acknowledgments</a></li>
</ul>

## Dataset Highlights

<b><i>360+x</i> dataset</b> introduces a unique panoptic perspective to scene understanding,
differentiating itself from existing datasets, by offering multiple viewpoints and modalities,
captured from a variety of scenes. Our dataset contains:


<ul>
<li><b>2,152 multi-model videos </b> captured by 360° cameras and Spectacles cameras (8,579k
    frames in total)

</li>
<li>
    <b>Capture in 17 cities</b> across 5 countries.
</li>
<li>
    <b>Capture in 28 Scenes</b> from Artistic Spaces to Natural Landscapes.
</li>
<li>
    <b>Temporal Activity Localisation Labels</b> for 38 action instances for each video.
</li>

</ul>

<img src="statics/overall.gif" />

## Dataset Access

The dataset is fully released in HuggingFace 🤗.

|                                      Low Resolution Version                                      |                                     High Resolution Version                                      |
|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| [Dataset quchenyuan/360x_dataset_LR](https://huggingface.co/datasets/quchenyuan/360x_dataset_LR) | [Dataset quchenyuan/360x_dataset_HR](https://huggingface.co/datasets/quchenyuan/360x_dataset_HR) |

The HuggingFace Repo also contains annotations.

## Toolkit Structure

- configs : Configuration files for the dataset
- libs : Libraries for the dataset
    - dataset : The dataloader for the dataset
    - database : The database for the dataset
- models : Models for the dataset

# Training

For using ActionFormer, you need to
follow [this compile guide](https://github.com/happyharrycn/actionformer_release/blob/main/INSTALL.md).

For training the model, you can use the following example script:

```bash
python run/TemporalAction/train.py \
       ./configs/tridet/360_i3d.yaml \
       --method tridet \
       --modality 10011
```

"run/TemporalAction/configs/tridet/360_i3d.yaml" is the configuration file for training.

Method identifies the model you want to train.

Modality is the input modality for the model. These five digits represent whether the model uses panoramic video,
front-view video, binocular video, audio, and direction audio respectively. For example, here "10011" means the model
uses panoramic video, audio, and direction audio.

# Pretrained Models

All pretrained models are available in
the [Huggingface Model Hub🤗](https://huggingface.co/quchenyuan/360x_dataset_pretrained_models).

| TAL Pretrained Model | mAP@0.50 | mAP@0.75 | mAP@0.95 |                                                 Download Link                                                 |
|:--------------------:|:--------:|:--------:|:--------:|:-------------------------------------------------------------------------------------------------------------:|
|     ActionFormer     |   27.4   |   17.0   |   6.53   | [Model](https://huggingface.co/quchenyuan/360x_dataset_pretrained_models/blob/main/TAL/actionformer.pth.tar)  |
|    TemporalMaxer     |   29.8   |   20.9   |   10.0   | [Model](https://huggingface.co/quchenyuan/360x_dataset_pretrained_models/blob/main/TAL/temporalmaxer.pth.tar) |
|        TriDet        |  26.98   |   19.4   |   7.21   |    [Model](https://huggingface.co/quchenyuan/360x_dataset_pretrained_models/blob/main/TAL/tridet.pth.tar)     |

For evaluation, you can use the following example script:

```bash
python run/TemporalAction/eval.py \
       ./configs/tridet/360_i3d.yaml \
       {path_to_pretrained_model.pth.tar} \
       --method tridet \
       --modality 10011
```


# Pretrained Models

Extracted features extracted by 360x pretrained extractor for each modality are also released in [Huggingface Dataset Hub🤗](https://huggingface.co/datasets/quchenyuan/360x_dataset_features)

## License

Distributed under the <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Cite

```
@inproceedings{chen2024x360,
  title={360+x: A Panoptic Multi-modal Scene Understanding Dataset},
  author={Chen, Hao and Hou, Yuqi and Qu, Chenyuan and Testini, Irene and Hong, Xiaohan and Jiao, Jianbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Contact

You can contact us by https://mix.jianbojiao.com/contact/.

You can also email us by <a href="mailto:mix.group.uk@gmail.com">mix.group.uk@gmail.com</a>
or <a href="mailto:cxq134@student.bham.ac.uk">cxq134@student.bham.ac.uk</a>.



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

This README template is inspired by [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


