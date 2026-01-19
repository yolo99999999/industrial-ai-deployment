"""
pytorch_basics
包含:张量操作、自动求导、简单神经网络构建
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def tensor_operations_demo():
    """
    1.张量基础操作
    """

    print("="*50)
    print("1.Pytorch张量基础操作")
    print("="*50)

    #1.1创建张量
    print("\n1.1创建张量:")
    #从列表创建
    tensor_a = torch.tensor([1,2,3,4,5])
    print(f"从列表创建:{tensor_a}")

    #创建全0张量
    zeros_tensor = torch.zeros(3,4)
    print(f"3x4全0张量:\n{zeros_tensor}")

    #创建全1张量
    ones_tensor = torch.ones(2,3)
    print(f"2x3全1张量:\n{ones_tensor}")

    #张量运算
    print("\n1.2张量运算")
    a = torch.tensor([[1,2],[3,4]],dtype=torch.float32)
    b = torch.tensor([[5,6],[7,8]],dtype=torch.float32)

    #加法
    print(f"a+b=\n{a+b}")

    #矩阵乘法
    print(f"a@b(矩阵乘法)=\n{a@b}")

    #元素乘法
    print(f"a*b(元素乘法)=\n{a*b}")

    #1.3张量形状操作
    print("\n1.3张量形状操作")
    tensor = torch.arange(12)
    print(f"原始张量:{tensor}")
    print(f"形状:{tensor.shape}")

    #重塑
    reshaped = tensor.reshape(3,4)
    print(f"重塑为3x4:\n{reshaped}")

    #转置
    transposed = reshaped.T
    print(f"转置:\n{transposed}")

    #1.4 与Numpy互操作
    print("\n1.4 与NumPy互操作:")
    np_array = np.array([1,2,3,4])
    torch_tensor = torch.from_numpy(np_array)
    print(f"NumPy数组:{np_array}")
    print(f"转换为Pytorch张量:{torch_tensor}")

    torch_to_np = torch_tensor.numpy()
    print(f"转换回NumPy:{torch_to_np}")

def autograd_demo():
    """
    2.自动求导
    """
    print("\n"+"="*50)
    print("2.PyTorch自动求导")
    print("="*50)

    #2.1 简单自动求导
    print("\n2.1 简单自动求导示例:")
    #创建一个需要梯度的张量
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    #定义一个计算
    z = x**2 + y**3 +x*y

    #计算梯度
    z.backward()

    print(f"x={x.item()}, y={y.item()}")
    print(f"z=x²+y²+xy={z.item()}")
    print(f"∂Z/∂x={x.grad.item()}")
    print(f"∂Z/∂y={x.grad.item()}")

    #2.2 神经网络中的自动求导
    print("\n2.2神经网络中的自动求导:")

    #定义一个简单的线性层
    class SimpleLinear(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(input_size,output_size))
            self.bias = nn.Parameter(torch.zeros(output_size))

        def forward(self, x):
            return x @ self.weights + self.bias
        
    #创建模型和输入
    model = SimpleLinear(3,2)
    input_tensor = torch.randn(4,3) #batch_size=4, input_size=3

    #向前传播
    output = model(input_tensor)

    #定义损失函数
    target = torch.randn(4,2)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)

    print(f"模型参数形状: weights={model.weights.shape}, bias={model.bias.shape}")
    print(f"输入形状:{input_tensor.shape}")
    print(f"输出形状:{output.shape}")
    print(f"损失值:{loss.item()}")

    #反向传播
    loss.backward()

    print(f"权重梯度形状:{model.weights.grad.shape}")
    print(f"偏置梯度形状:{model.bias.grad.shape}")

    return model

def simple_nn_demo():
    """
    3.简单神经网络构建
    """
    
    print("\n"+"="*50)
    print("3.简单神经网络构建")
    print("="*50)

    #3.1 构建一个简单的三层全连接网络
    print("\n3.1 构建全连接神经网络:")

    class SimpleNN(nn.Module):
        """一个简单的三层全连接网络"""
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_size, hidden_size//2)
            self.layer3 = nn.Linear(hidden_size//2, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer3(x)
            x = self.sigmoid(x)
            return x
        
    #创建网络实例
    model = SimpleNN(input_size=10, hidden_size=64, output_size=2)
    print(f"网络结构:\n{model}")

    #打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n参数统计:")
    print(f"总参数:{total_params:,}") 
    print(f"可训练参数:{trainable_params:,}")

    #3.2 测试网络向前传播
    print("\n3.2 测试向前传播:")
    test_input = torch.randn(5,10)  # batch_size=5, input_size=10
    output = model(test_input)

    print(f"输入形状:{test_input.shape}")
    print(f"输出形状:{output.shape}")
    print(f"输出值范围:[{output.min().item():.4f},{output.max().item():.4f}]")

    return model
def linear_regression_demo():
    """
    4.线性回归实战演示(与sklearn对比)
    """

    print("\n"+"="*50)
    print("4.线性回归实战")
    print("="*50)

    #4.1 创建模拟数据
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 200
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.rand(n_samples, 1)

    #转换为PyTorch张量
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    #4.2 使用sklearn线性回归(作为基准)
    from sklearn.linear_model import LinearRegression
    lr_sklearn = LinearRegression()
    lr_sklearn.fit(X,y)

    sklearn_intercept = lr_sklearn.intercept_[0]
    sklearn_coef = lr_sklearn.coef_[0][0]

    print(f"Sklearn线性回归结果:")
    print(f"截距(bias):{sklearn_intercept:.4f}")
    print(f"系数(weight):{sklearn_coef:.4f}")

    #4.3 使用PyTorch手动实现线性回归
    print("\n4.3 PyTorch手动实现线性回归:")

    #定义模型
    class ManualLinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(1,1))
            self.bias = nn.Parameter(torch.randn(1))

        def forward(self, x):
            return x @ self.weight+self.bias
        
    model = ManualLinearRegression()

    #定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #训练模型
    n_epochs = 1000
    losses = []

    for epoch in range(n_epochs):
        #向前传播
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        #每训练到第 200 的整数倍轮次时，打印一次当前损失
        if (epoch+1)%200 == 0:        
            print(f"Epoch[{epoch+1}/{n_epochs}],Loss:{loss.item():.4f}")

        #获取训练后的参数
    trained_weight = model.weight.item()
    trained_bias = model.bias.item()

    print(f"\nPyTorch线性回归结果:")
    print(f" 截距 (bias): {trained_bias:.4f}")
    print(f" 系数 (weight): {trained_weight:.4f}")

    #4.4 可视化结果
    plt.figure(figsize=(12,4))

    #子图1:原始数据和拟合线
    plt.subplot(1,2,1)
    plt.scatter(X, y, alpha=0.7, label="原始数据")

    #Sklearn拟合线
    X_range = np.array([[0],[2]])
    y_sklearn = lr_sklearn.predict(X_range)
    plt.plot(X_range, y_sklearn, "r-", linewidth=3, label="Sklearn拟合")

    #PyTorch拟合线
    X_range_tensor = torch.from_numpy(X_range).float()
    y_pytorch = model(X_range_tensor).detach().numpy()
    plt.plot(X_range, y_pytorch, "g--", linewidth=3,label="PyTorch拟合")

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归拟合比较')
    plt.legend()   #把当前坐标系里所有带 label= 的曲线/散点/柱形自动收集起来，画一个图例框
    plt.grid(True, alpha=0.3)

    #子图2:损失下降曲线
    plt.subplot(1,2,2)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('训练损失下降曲线')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('linear_regression_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存为:linear_regression_comparison.png")

    return model, losses
    
def binary_classification_demo():
    """
    5. 二分类问题实战
    """

    print("\n"+"="*50)
    print("5. 二分类问题实战")
    print("="*50)        

    #5.1 创建非线性可分的二分类数据
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

    #转换为PyTorch张量
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().unsqueeze(1) #添加维度

    #划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

    print(f"数据集形状:X={X.shape}, y={y.shape}")
    print(f"训练集:{X_train.shape[0]}个样本")
    print(f"测试集:{X_test.shape[0]}个样本")

    #5.2 构建分类模型
    class ClassificationModel(nn.Module):
        def __init__(self, input_size=2, hidden_size=16, output_size=1):
            super().__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(hidden_size, hidden_size)
            self.relu2 = nn.ReLU()
            self.layer3 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.relu1(x)
            x = self.layer2(x)
            x = self.relu2(x)
            x = self.layer3(x)
            x = self.sigmoid(x)
            return x
        
    model = ClassificationModel()

    #5.3 训练模型
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)

    n_epochs= 500
    train_losses = []
    test_losses = []
    accuracies = []

    print("\n开始训练模型...")
    for epoch in range(n_epochs):
        #训练阶段
        model.train()
        train_predictions = model(X_train)
        train_loss = criterion(train_predictions, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        #评估阶段
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)

            #计算准确率
            predicted_labels = (test_predictions>0.5).float()
            accuracy = (predicted_labels==y_test).float().mean()

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        accuracies.append(accuracy.item())

        if (epoch+1)%100 == 0:
            print(f"Epoch[{epoch+1}/{n_epochs}],"
                f"Train Loss:{train_loss.item():.4f},"
                f"Test Loss:{test_loss.item():.4f},"
                f"Accuracy:{accuracy.item():.4f}")
            
    #5.4 可视化决策边界
    plt.figure(figsize=(15,5))

    #子图1:原始数据分布
    plt.subplot(1,3,1)
    plt.scatter(X[y==0,0], X[y==0,1], c='blue', alpha=0.7, label='类别0')
    plt.scatter(X[y==1,0], X[y==1,1], c='red', alpha=0.7, label='类别1')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('原始数据分布(moons数据集)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    #子图2:训练曲线
    plt.subplot(1,3,2)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    #子图3:决策边界
    plt.subplot(1,3,3)

    #创建网格点
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max,0.1),
                        np.arange(y_min, y_max,0.1))
    
    #预测网格点
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    with torch.no_grad():
        Z = model(grid_tensor)
        Z = (Z > 0.5).float().numpy()

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[y==0,0], X[y==0,1], c='blue', alpha=0.7, label='类别0')
    plt.scatter(X[y==1,0], X[y==1,1], c='red', alpha=0.7, label='类别1')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('模型决策边界')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('binary_classification_results.png', dpi=150, bbox_inches='tight')
    print(f"\n分类结果已保存为:binary_classification_results.png")

    #最终阶段
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_test)
        final_accuracy = ((final_predictions>0.5).float()==y_test).float().mean()
    print(f"\n最终测试准确率:{final_accuracy.item():.4f}")

    return model, final_accuracy.item()

def main():
    """
    主函数:按顺序执行所有示例
    """

    print("PyTorch基础学习脚本")
    print("="*60)

    try:
        #1.张量操作
        tensor_operations_demo()

        #2.自动求导
        autograd_model = autograd_demo()

        #3.简单神经网络构建
        simple_nn_model = simple_nn_demo()

        #4.线性回归
        linear_model, linear_losses = linear_regression_demo()

        #5.二分类问题
        classification_model, accuracy = binary_classification_demo()

        print("\n" + "="*60)
        print("Day1 学习完成!")
        print("已掌握:")
        print("1.PyTorch张量基础操作")
        print("2.自动求导机制")
        print("3.简单神经网络构建")
        print("4.线性回归实现")
        print("5.二分类问题解决")
        print("="*60)

        return{
            'autograd_model':autograd_model,
            'simple_nn_model':simple_nn_model,
            'linear_model':linear_model,
            'classification_model':classification_model,
            'linear_losses':linear_losses,
            'accuracy':accuracy
        }
    
    except Exception as e:
        print(f"执行过程中出现错误:{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    results = main()
        


                  


        



    



    


