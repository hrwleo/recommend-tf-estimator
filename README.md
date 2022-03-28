# recommend-tf-estimator


[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-1.4+/2.0+-blue.svg)](https://pypi.org/project/deepctr)

![1](https://img.shields.io/badge/Python-3.x-brightgreen.svg)

## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   DeepFM      | [arXiv 2017] [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)                                       |
|   ESMM        | [SIGIR 2018] [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)                    |
|   MMOE        | [KDD 2018] [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007)                 |
|   FiBiNET     | [RecSys 2019] [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)     |
|   TwoTower    | [arXiv 2020] [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)                                                                      |

## code sturcture

```
--config            训练配置，可根据业务新增 
--data              数据样本 
--src 
    --input_fn      输入相关函数 
    --models_ompl   模型实现
    --common_utils  通用工具函数，包含特殊层，特殊loss的实现
--examples          运行样例
--online_deploy     部署脚本
--test              测试脚本
```

```
--common_utils  
layers                  特殊层实现(SENet, 双线性交叉层, attention层等)
loss_fn                 损失函数
wpai_model_auto_update  更新在线预测的模型

```

```
--layers
dice
prelu
build_deep_layers
build_Bilinear_Interaction_layers
build_SENET_layers
attention_layer
batch_norm_layer
```

## quick start
```
cd examples
python train_esmm.py

* mmoe 实现了base和wide+esmm版本
使用方法：
在初始化estimaor时，指定model_fn即可
```


