import torch
import matplotlib.pyplot as plt
import net

# 假设已经实例化了改进的AOD-Net模型
model = net.AODnet_ours()

# 定义用于提取特征图的钩子函数
def get_activation(name):
    def hook(model, input, output):
        setattr(model, name, output.detach())
    return hook

# 注册钩子函数以获取特定层的输出
activation = {}
model.conv2_1.register_forward_hook(get_activation('conv2_1_output'))

# 通过前向传播获取输出
input_data = torch.randn(1, 3, 640, 480)  # 示例输入数据
output = model(input_data)

# 获取特定层的输出
conv2_1_output = activation['conv2_1_output']

# 可视化特征图
plt.imshow(conv2_1_output[0, 0].cpu().numpy(), cmap='hot', interpolation='nearest')
plt.title('Channel-wise Feature Map')
plt.colorbar()
plt.show()
