# Models Directory

此目录包含Bridge Tuning实验的所有模型定义。

## 模型结构

### 1. Foundation Models (基础模型)
- `swin_unetr.py`: Swin UNETR模型（在TotalSegmentator上预训练）
- 其他全监督医学图像分割模型

### 2. Bridge Models (桥接模型)
- `bridge_model.py`: 桥接微调模型
  - 使用教师-学生架构
  - 集成PEFT方法（LoRA等）
  - 支持知识蒸馏

### 3. PEFT Modules (参数高效微调模块)
- `peft_modules/lora.py`: LoRA实现
- `peft_modules/adapter.py`: Adapter实现（可选）
- `peft_modules/prefix_tuning.py`: Prefix Tuning实现（可选）

## 待实现

### Bridge Model设计思路

桥接模型需要实现以下功能：

1. **教师-学生架构**
   - 教师模型：冻结的预训练Foundation Model
   - 学生模型：带PEFT的轻量化模型

2. **PEFT集成**
   - 在关键层插入LoRA模块
   - 只训练LoRA参数，保持backbone冻结
   - 参数量控制在总参数的1-5%

3. **知识蒸馏**
   - 特征层面的蒸馏
   - 输出层面的蒸馏（soft targets）
   - 损失函数组合

4. **模型结构示例**
```python
class BridgeModel(nn.Module):
    def __init__(self, teacher_model, peft_config):
        self.teacher = teacher_model  # 冻结
        self.student = self._build_student_with_peft()

    def forward(self, x):
        # 教师模型推理（用于蒸馏）
        with torch.no_grad():
            teacher_features, teacher_output = self.teacher(x)

        # 学生模型推理
        student_features, student_output = self.student(x)

        return {
            'output': student_output,
            'teacher_output': teacher_output,
            'student_features': student_features,
            'teacher_features': teacher_features
        }
```

## 使用说明

模型需要在`__init__.py`中注册后才能被训练器使用。

```python
from .swin_unetr import SwinUNETR
from .bridge_model import BridgeModel

__all__ = ['SwinUNETR', 'BridgeModel']
```
