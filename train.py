import torch
import torch.nn as nn
import torch.optim as optim
from model import ResNet18, Residual
from dataset import get_dataloaders
import time

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========================================")
    print(f"开始训练！使用的设备: {device}")
    print(f"=========================================")
    
    epochs = 10
    batch_size = 128
    learning_rate = 0.001
    data_dir = './data/PetImages'
    
    train_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size)
    print(f"训练集批次数: {len(train_loader)} | 测试集批次数: {len(test_loader)}")
    
    model = ResNet18(Residual).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
                
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
        end_time = time.time()
        print(f"\n---> Epoch {epoch+1} 结束 | 耗时: {end_time - start_time:.0f}秒 | 测试集准确率: {100.*test_correct/test_total:.2f}%\n")
        
    torch.save(model.state_dict(), "resnet18_catdog_scratch.pth")
    print("🎉 训练完成！模型权重已保存为 resnet18_catdog_scratch.pth")

if __name__ == '__main__':
    train()

    