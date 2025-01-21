# 实验五 多模态情感预测

本次实验通过给定配对的文本和图像，结合两者信息，使用多模态模型，预测在测试集上图片对应的情感标签，情感标签共分为三种：`negative`, `neutral`, `positive`。

## Setup
```
torch==1.9.0
transformers==4.5.0
matplotlib==3.4.0
pandas==1.2.0
scikit-learn==0.24.0
tqdm==4.59.0
nltk==3.9.1
```
可以直接在命令行安装依赖：
```bash
pip install -r requirements.txt
```


## Repository Structure

```
project/
├── config.py               # 配置文件，包含模型参数和路径设置
├── data_processing.py      # 数据加载，数据训练与预测前的预处理
├── model.py            # 定义了BERT、ResNet和多模态融合模型
├── train.py            # 训练脚本，支持仅文本、仅图像和多模态模型的训练
├── test.py             # 测试脚本，预测不同模型下的结果
├── evaluate.py         # 生成多模态融合模型预测结果
├── data/                       # 数据集，需要自行下载
│   ├── train.txt               # 训练数据
│   ├── test_without_label.txt  # 测试数据
│   └── image_data/             # 图像数据
├── result/                     # 结果目录，保存模型和预测结果
│   ├── predictions_with_ablation.txt   # 消融实验预测结果
│   └── predictions_best.txt    # 多模态融合模型的预测结果
├── 10225501446_闫研_实验五.pdf  # 实验报告
└── requirements.txt            # 本次实验所需环境
```

## Usage
本次实验推荐使用GPU环境进行训练。

实验中涉及的预训练模型BERT和ResNet模型需要事先下载，直接运行`train.py`即可，下载时注意网络环境。

实验前，将所有文件放在根目录下，提前将data下载到同一目录。
1. 在根目录下创建`data`目录，将训练数据`train.txt`和测试数据`test_without_label.txt`放置在`data`目录下。
2. 将图像和对应的文件放置在`data/image_data`目录下。

训练前可在config.py中调试需要使用的参数。

在终端运行以下命令训练模型:

```bash
python train.py
```

模型会根据验证集的表现自动保存最佳模型到 `result` 目录。

运行以下命令进行模型预测，结果保存到 `result/predictions.txt`。：

```bash
python evaluate.py
```


## Reference
1. https://github.com/RecklessRonan/GloGNN/blob/master/readme.md
2. https://github.com/smartcameras/SelfCrossAttn
3. T. Zhu, L. Li, J. Yang, S. Zhao, H. Liu and J. Qian, "Multimodal Sentiment Analysis With Image-Text Interaction Network," in IEEE Transactions on Multimedia, vol. 25, pp. 3375-3385, 2023, doi: 10.1109/TMM.2022.3160060.
4. https://blog.csdn.net/weixin_44211968/article/details/120995096?spm=1001.2014.3001.5502