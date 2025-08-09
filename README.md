# HyperKD: Lifelong Hyperspectral Image Classification With Cross-Spectral–Spatial Knowledge Distillation

## Environment
PyTorch 2.1.1, CUDA 11.8, and Ubuntu 20.04

## Run commands
### Indian Pines
python3 main.py --config configs/our_indian.json
### Houston 2013
python3 main.py --config configs/our_houston.json
### Salinas
python3 main.py --config configs/our_salinas.json


## If you find this work helpful, please cite:
@ARTICLE{11023858,
  author={Li, Zhenlin and Xia, Shaobo and Wang, Shuhe and Yue, Jun and Fang, Leyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Exemplar-Free Lifelong Hyperspectral Image Classification With Spectral Consistency}, 
  year={2025},
  volume={63},
  number={},
  pages={1-16},
  keywords={Computational modeling;Adaptation models;Data models;Training;Training data;Predictive models;Incremental learning;Generators;Prototypes;Image reconstruction;Hyperspectral image (HSI) classification;incremental learning;lifelong learning;model inversion;spectral consistency},
  doi={10.1109/TGRS.2025.3576643}}


## Our project references the codes in the following repositories.
[ADC](https://github.com/dipamgoswami/ADC)

