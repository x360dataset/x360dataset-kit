
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
- [ ] Checkpoints Release


### Table of Contents
<ul>
<li>
  <a href="#about-the-project">About The Project</a>
  <ul>
    <li><a href="#built-with">Built With</a></li>
  </ul>
</li>
<li>
  <a href="#getting-started">Getting Started</a>
  <ul>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#installation">Installation</a></li>
  </ul>
</li>
<li><a href="#usage">Usage</a></li>
<li><a href="#roadmap">Roadmap</a></li>
<li><a href="#contributing">Contributing</a></li>
<li><a href="#license">License</a></li>
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


## Toolkit Structure

- configs : Configuration files for the dataset
- libs : Libraries for the dataset
  - dataset : The dataloader for the dataset
  - database : The database for the dataset
- models : Models for the dataset


## License

Distributed under the <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.



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

You can also email us by <a href="mailto:mix.group.uk@gmail.com">mix.group.uk@gmail.com</a> or <a href="mailto:cxq134@student.bham.ac.uk">cxq134@student.bham.ac.uk</a>.



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This README template is inspired by [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


