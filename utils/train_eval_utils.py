import torch
import numpy as np
import random
import torch.nn as nn # 导入 nn 模块以便在函数中使用
import torch.optim as optim # 导入 optim 模块

def set_seed(seed_value):
    """
    固定所有随机性，以确保实验的可复现性。
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.backends.mps.is_available(): # For Apple Silicon MPS
        torch.mps.manual_seed(seed_value)
    elif torch.cuda.is_available(): # For NVIDIA GPUs
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # 确保CUDA操作是确定性的，可能会略微降低性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value}")

def train_model(model, train_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    """
    执行模型的训练循环。

    Args:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数。
        optimizer (optim.Optimizer): 优化器。
        num_epochs (int): 训练的轮次。
        device (torch.device): 训练设备 (CPU/MPS)。
        scheduler (optional): 学习率调度器。
    """
    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # 清零梯度
            outputs = model(images) # 前向传播
            
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = 100 * correct_predictions / total_samples
        
        # 如果有调度器，更新学习率
        if scheduler:
            scheduler.step() # 每完成一个epoch后，调度器更新学习率
            current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, Current LR: {current_lr:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%")

    print("--- Training Finished ---")

def evaluate_model(model, test_loader, device, classes=None):
    """
    评估模型在测试集上的性能。

    Args:
        model (nn.Module): 要评估的模型。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 评估设备 (CPU/MPS)。
        classes (tuple, optional): 数据集类别名称，用于打印。
    """
    print("\n--- Starting Evaluation ---")
    model.eval() # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the {total} test images: {accuracy:.2f}%")
    print("--- Evaluation Finished ---")
    return accuracy

def save_model(model, path):
    """
    保存模型的 state_dict。
    """
    torch.save(model.state_dict(), path)
    print(f"\nModel saved to {path}")

def load_model(model_class, path, device):
    """
    加载模型的 state_dict。
    """
    loaded_model = model_class().to(device) # 实例化一个新的模型
    loaded_model.load_state_dict(torch.load(path)) # 加载保存的参数
    loaded_model.eval() # 加载后通常立即设置为评估模式
    print(f"\nModel loaded successfully from {path}")
    return loaded_model