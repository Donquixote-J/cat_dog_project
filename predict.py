import torch
from PIL import Image
from torchvision import transforms
from model import ResNet18, Residual
import torch.nn.functional as F

def predict_image(image_path, model_path="resnet18_catdog_scratch.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化模型并加载你刚训练好的权重
    model = ResNet18(Residual).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 成功加载模型权重！")
    except Exception as e:
        print(f"❌ 加载权重失败，请检查文件是否存在: {e}")
        return
        
    model.eval() # 切换到预测（评估）模式
    
    # 2. 图像预处理（必须和训练时一模一样）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. 读取并处理传入的图片
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ 无法打开图片 {image_path}: {e}")
        return

    # 增加一个 batch 维度 (模型需要 [batch, channels, height, width] 格式的输入)
    img_tensor = transform(img).unsqueeze(0).to(device) 
    
    # 4. 进行预测
    with torch.no_grad():
        output = model(img_tensor)
        # 将输出转换为百分比概率
        probabilities = F.softmax(output, dim=1)[0]
        
    # 5. 解析结果 (我们在 dataset.py 里设定的：0是猫，1是狗)
    cat_prob = probabilities[0].item() * 100
    dog_prob = probabilities[1].item() * 100
    
    print("-" * 30)
    if cat_prob > dog_prob:
        print(f"🐱 预测结果: 这是一个 猫！ (置信度: {cat_prob:.2f}%)")
    else:
        print(f"🐶 预测结果: 这是一个 狗！ (置信度: {dog_prob:.2f}%)")
    print("-" * 30)

if __name__ == "__main__":
    # 我们先随便挑一张数据集里的狗狗图片来测试
    test_img = "./data/PetImages/Dog/100.jpg" 
    print(f"正在识别图片: {test_img}")
    predict_image(test_img)
