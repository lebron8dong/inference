import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import torch.nn.functional as F
import time

def preprocess_image(image_path):
    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 打开图片
    image = Image.open(image_path).convert('RGB')

    # 应用预处理
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor


# 创建模型实例
model = resnet50()

# 加载预训练的权重
model.load_state_dict(torch.load('resnet50.pth'))

# 设置为评估模式
model.eval()

# 调用预处理函数处理您的图片
input_tensor = preprocess_image('cat.jpg')

# 使用模型进行预测
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]
    
    # 找到最大概率的索引
    max_prob_index = torch.argmax(probabilities).item()
    max_prob = probabilities[max_prob_index].item()

    # 输出最大概率的预测结果
    print(f"The image is predicted to be class {max_prob_index} with a probability of {max_prob:.2f}.")
time.sleep(20)
print("stop")