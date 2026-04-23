import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 超参数设置 =====================
INPUT_DIM = 10  # 输入向量的维度（可自由修改：5/20/50都可以）
NUM_CLASSES = INPUT_DIM  # 类别数 = 向量维度（最大值索引就是类别）
BATCH_SIZE = 32  # 批次大小
EPOCHS = 20  # 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 2. 生成数据集 =====================
def generate_data(num_samples):
    """
    生成训练/测试数据：
    输入：随机向量 (num_samples, INPUT_DIM)
    标签：向量中最大值的索引 (num_samples,)
    """
    # 生成随机数据（正态分布随机数）
    data = torch.randn(num_samples, INPUT_DIM)
    # 生成标签：取每一行最大值的索引
    labels = torch.argmax(data, dim=1)
    return data, labels


# 生成训练集（10000个样本）和测试集（2000个样本）
train_data, train_labels = generate_data(10000)
test_data, test_labels = generate_data(2000)

# 包装成数据加载器（批量训练）
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ===================== 3. 构建神经网络模型 =====================
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        # 简单全连接网络：输入层 → 隐藏层 → 输出层
        self.fc1 = nn.Linear(input_dim, 64)  # 第一层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(64, num_classes)  # 输出层（输出=类别数）

    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 初始化模型、损失函数、优化器
model = Classifier(INPUT_DIM, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()  # 多分类任务标准损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# ===================== 4. 训练模型 =====================
print("开始训练...")
for epoch in range(EPOCHS):
    model.train()  # 训练模式
    train_loss = 0.0
    correct = 0
    total = 0

    for data, labels in train_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)

        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(data)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播 + 更新参数
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 打印训练信息
    avg_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{EPOCHS}] | 损失: {avg_loss:.4f} | 训练准确率: {acc:.2f}%")

# ===================== 5. 测试模型 =====================
print("\n开始测试...")
model.eval()  # 测试模式
with torch.no_grad():  # 关闭梯度计算
    correct = 0
    total = 0
    for data, labels in test_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"测试集准确率: {100 * correct / total:.2f}%")

# ===================== 6. 单样本预测（验证效果） =====================
print("\n单样本预测测试：")
# 生成一个随机向量
test_vector = torch.randn(1, INPUT_DIM).to(DEVICE)
# 模型预测
model.eval()
with torch.no_grad():
    output = model(test_vector)
    pred_class = torch.argmax(output, dim=1).item()
    true_class = torch.argmax(test_vector, dim=1).item()

print(f"输入随机向量: {test_vector.cpu().numpy()[0]}")
print(f"真实类别(最大值索引): {true_class}")
print(f"模型预测类别: {pred_class}")