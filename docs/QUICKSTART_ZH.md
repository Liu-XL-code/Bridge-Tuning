# 中文使用说明：纯源码版

本包不包含任何预训练或已训练权重，也不自动下载权重。包含 Bridge-Tuning 的桥域训练、
目标域少样本微调、独立验证、测试和原始空间三维推理；没有 DA 对比方法代码。

## 1. 安装

推荐 Python 3.10/3.11。先按 PyTorch 官网安装与 GPU 匹配的版本，再执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

已验证的版本见 `requirements-verified.txt`。完整模型训练建议使用 CUDA GPU。

## 2. 不下载权重，先跑通训练、验证、测试和推理

```bash
python run_smoke.py --output runs/smoke --device auto
python -m unittest discover -s tests -v
python infer.py --checkpoint runs/smoke/target/selected.pt \
  --image examples/synthetic/images/synthetic_10.nii.gz \
  --output runs/demo/pred.nii.gz --device auto
python visualize.py --image examples/synthetic/images/synthetic_10.nii.gz \
  --label examples/synthetic/labels/synthetic_10.nii.gz \
  --prediction runs/demo/pred.nii.gz --output runs/demo/montage.png --synthetic
```

这会在本地生成模型权重，先完成两轮桥域训练和两轮目标微调，再评估独立验证集和测试集。
`verification.json` 记录训练是否读取测试集、输出是否恢复原始尺寸和 affine。
示例采用 feature_size=12、64³ ROI；论文模型为 feature_size=48、96³ ROI。
合成示例用于软件检查，不能代表临床性能。每次训练请使用新的输出目录。

## 3. 准备自己的基础模型

正式方法需要另行取得兼容的基础权重，来源和格式见 `MODEL_CARD.md`。
对可信来源的旧 checkpoint 执行一次转换：

```bash
python convert_checkpoint.py --input weights/foundation_swinunetr.pth \
  --config configs/ct_bridge.yaml --role foundation --backbone-only --trusted \
  --output runs/foundation.pt
```

转换会移除旧输出头；初始化会逐项检查非输出头参数是否兼容。基础 backbone 不能直接
用于肿瘤二分类推理。不要用随机初始化的调试结果替代论文的预训练方法。

## 4. 使用自己的数据

每行对应一个三维影像及同网格标签，路径相对 CSV：

```csv
id,subject_id,image,label
case001,subject001,images/case001.nii.gz,labels/case001.nii.gz
```

`subject_id` 防止同一患者的重复扫描或心动周期帧跨集合。先检查几何关系：

```bash
python audit_geometry.py --csv data/train.csv --output runs/geometry.json
```

代码不修改原始图像或标签。尺寸或 affine 不一致时终止并报告匿名编号，需先确认空间对齐。

## 5. 正式桥域训练与 K-shot 目标微调

```bash
python train.py --config configs/ct_bridge.yaml --train-csv data/bridge_train.csv \
  --val-csv data/bridge_val.csv --init runs/foundation.pt \
  --role bridge --output runs/ct_bridge --device auto
python train.py --config configs/ct_target_lora.yaml --train-csv data/target_train.csv \
  --init runs/ct_bridge/selected.pt --role target --seed 0 \
  --output runs/ct_target --device auto
python evaluate.py --checkpoint runs/ct_target/selected.pt --csv data/target_test.csv \
  --partition test --output runs/test --save-predictions --device auto
```

独立验证可用 `evaluate.py --partition validation`，只评估固定模型。
桥域可显式设置 `training.selection: val_dice` 并提供不重叠的验证集；
目标域没有验证集时选择 `final`，不能用测试集选模型。
目标微调默认继承桥域二分类头；历史运行若重置头，需要显式传入 `--reset-head`。
交叉验证命令见英文 README 和 `TRAINING.md`。

## 6. 预处理与统计

CT：RAS → 1.5×1.5×2.0 mm → HU [-175,250] 截断并缩放到 [0,1] → 图像正值前景裁剪；
训练时补齐、抽取 96³ patch，并执行空间和强度增强。标签使用最近邻插值。
MRI：RAS → 1×1×1.5 mm → 0.5/99.5 百分位缩放 → 前景裁剪，不使用 CT HU 窗。
M&Ms、CC-359、SAML 的标签规则不同，应使用对应配置，详见 `PREPROCESSING.md`。

`metrics.json` 的标准差是病例间样本标准差；少于两例时为 `null`。
`aggregate_results.py` 先在每个种子内平均各折，再计算三个种子的均值和标准差。
旧论文的 4/5 折×3 种子、缺失记录和历史过滤行为另见 `RESULTS.md`。
本版不能仅凭种子编号重建未保留的历史划分，也没有重新训练论文的全部临床实验。

## 7. 文件核验

```bash
python verify_files.py
```

`DATASETS.md` 给出数据来源；`TRAINING.md` 给出参数、划分、停止和选模规则；
`MODEL_CARD.md` 说明外部权重和自行生成 checkpoint 的用法；`VERIFICATION.md` 记录实际检查。
`examples/synthetic/` 的十二对影像和标签完全合成，未包含患者影像。
