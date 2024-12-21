import torch
import torchvision.models as models  # 或者导入你的自定义模型
import torch.nn as nn

# 假设你使用的是 ResNet18 模型
model = models.resnet18()  # 用你实际的模型替换
model.fc = nn.Linear(model.fc.in_features, 10)  # 改为适配10类
model.load_state_dict(torch.load('/home/featurize/work/RAI2project/RAI2checkpoint/similarity/cifar10/resnet18/vic/model_0.pth'), strict=False)
model.eval()  # 设置为评估模式

# 检查模型结构
print(model)
