# CL2P

PyTorch implementation for the paper "[Disentangling Multi-view Representations via Curriculum Learning with Learnable Prior](https://cshaowang.github.io/files/IJCAI2025_CL2P.pdf)" (IJCAI 2025)

### Table of contents
* [Introduction](#star2-introduction)
* [Roadmap](#compass-roadmap)
* [Installation](#gear-installation)
* [Dataset and model](#dart-dataset-and-model)
* [Usage](#zap-usage) 
* [Contact](#raising_hand-questions)


## :star2: Introduction
<table>
  <tr>
    <td width="50%">
    This work reveals a coupling between neural networks’ “easy-to-hard” preference in learning and the trade-off of view consistency vs. specificity.
    </td>
    <td width="50%">
    <img src="figures/main-idea.png" width="800">
    </td>
  </tr>
</table>

<!-- ![framework](figures/main-idea.png) -->
![framework](figures/framework.png)


## 	:compass: Roadmap
This is our code structure:
```
CL2P/
├── configs/ # Store the trained dPoE.
├── data/ # Store the proposed modules.
|	├── datatool.py  ===  The proposed dPoE model
|	└── dPoeTraining.py  === Train the proposed model
├── models/ # Datasets, data processing, model loader, and evalaution.
|	├── autoencoder.py # Store the datasets
|	├── consistency_models.py === Load data, and generate anomalies
|	├── ms_estimators.py  === Performance evaluation using AUC
|	├── model.py  === Performance evaluation using AUC
|	└── specificity_models.py  === Load the trained model
├── utils/ # Datasets, data processing, model loader, and evalaution.
|	├── metrics.py ===
|	├── misc.py === Load data, and generate anomalies
|	├── optimizer.py  === Performance evaluation using AUC
|	└── visualization.py  === Load the trained model
├── README.md === THIS file!
├── test.py === run (training and test) dPoE
└── train.py === THIS file!

```

## :gear: Installation
* python == 3.10.15
* torch == 2.1.0
* torchvision == 0.16.0
* scikit-learn == 1.5.2
* scipy == 1.14.1

We also export our conda virtual environment as CL2P.yaml. You can use the following command to create the environment.
```bash
conda env create -f CL2P.yaml
```

## :dart: Dataset and model
You could find the Office-31 dataset we used in the paper from [Baidu Netdisk](https://pan.baidu.com/s/1lcE6gEwuO0k1nR_m17gtKw?pwd=hwvx), and the pre-trained models from [Baidu Netdisk](https://pan.baidu.com/s/10FRHrgtLhAE08ENblP4vsg?pwd=3utf).

## :zap: Usage
### Training
To train the model, use the following command:

```bash
python train.py -f configs/Edge-MNIST.yaml
```
This will start the training process using the configuration specified in `configs/Edge-MNIST.yaml`.

### Testing
To test the trained model, use the following command:
```bash
python test.py -f configs/Edge-MNIST.yaml
```
This will load the trained model and test it using the configuration specified in `configs/Edge-MNIST.yaml`.

## :raising_hand: Questions
Should you have any questions, please feel free to contact me (kaiguo.gm@gmail.com) or open an issue.

## :books: Citation
If you find CL2P useful in your research, please consider giving it a ⭐️ and citing:
```latex
@inproceedings{guo2025disentangling,
  title={Disentangling Multi-view Representations via Curriculum Learning with Learnable Prior},
  author={Guo, Kai and Wang, Jiedong and Peng, Xi and Hu, Peng and Wang, Hao},
  journal={Proceedings of the 34th International Joint Conference on Artificial Intelligence},
  year={2025}
}
```

## :sparkles: Acknowledgements
The codes are based on [MRDD](https://github.com/Guanzhou-Ke/MRDD). Thanks to the authors for their codes!
