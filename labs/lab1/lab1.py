import torch
import torch.nn as nn
import torch.nn.functional as F

# สร้างเลเยอร์ไว้ภายนอกฟังก์ชัน
conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=4, stride=2, padding=0)
conv2 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=3, stride=1, padding=0)
pool = nn.MaxPool2d(kernel_size=2)
leaky_relu = nn.LeakyReLU()

# สร้างฟังก์ชันสำหรับ forward pass
def forward_pass(x):
    print("Input:", x.shape)

    x = conv1(x)
    print("After Conv1:", x.shape)

    x = F.relu(x)
    print("After ReLU:", x.shape)

    x = pool(x)
    print("After MaxPool1:", x.shape)

    x = conv2(x)
    print("After Conv2:", x.shape)

    x = leaky_relu(x)
    print("After LeakyReLU:", x.shape)

    x = pool(x)
    print("After MaxPool2:", x.shape)

    return x

# สร้าง input และเรียกใช้งานฟังก์ชัน
x_input = torch.randn(1, 1, 100, 100)
output = forward_pass(x_input)

print("Final Output Tensor Size:", output.shape)
