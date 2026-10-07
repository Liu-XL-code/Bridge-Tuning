# Networks Directory

此目录包含各种网络架构的实现。

## 目录结构

- `backbones/`: 各种backbone网络（Swin Transformer, ResNet等）
- `encoders/`: 编码器模块
- `decoders/`: 解码器模块
- `blocks/`: 基础网络模块（Attention, Conv blocks等）

## 说明

网络架构和模型的区别：
- Networks: 底层网络结构组件（如Swin Transformer的各个模块）
- Models: 完整的模型定义（如SwinUNETR整体架构）

模型会调用networks中的组件来构建完整的网络。
