import torch
import torch.nn as nn
# import torch.nn.functional as F
# from models.sequence_modeling import BidirectionalLSTM
# from sequence_modeling import BidirectionalLSTM


class G_Net(nn.Module):
    def __init__(self, n_class =2, input_size=112,dropout_factor=0.5):
        super(G_Net, self).__init__()

        # conv1_out = cfg.conv1_out
        # conv2_out = cfg.conv2_out
        # conv3_out = cfg.conv3_out
        # fc1_out = cfg.fc1_out
        # fc2_out = cfg.fc2_out

        # self.conv1 = nn.Conv2d(in_channels =3 , out_channels = 8 , kernel_size = 3,stride=2)
        # self.bn1 = nn.BatchNorm2d(8, affine=True)
        # self.relu6 = nn.ReLU(inplace=True)
        # self.depthwise1 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, stride=1,padding=1, groups=8)
        # self.bn2 = nn.BatchNorm2d(8, affine=True)
        # self.conv2 = nn.Conv2d(in_channels =8 , out_channels = 16 , kernel_size = 3,stride=1,padding=1)
        # self.bn3 = nn.BatchNorm2d(16, affine=True)
        # self.depthwise2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, stride=1,padding=1, groups=16)
        # self.bn4 = nn.BatchNorm2d(16, affine=True)
        # self.conv3 = nn.Conv2d(in_channels =16 , out_channels = 8 , kernel_size = 3,stride=1,padding=1)
        # self.bn5 = nn.BatchNorm2d(8, affine=True)
        # self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3)
        # self.bn2 = nn.BatchNorm2d(conv2_out, affine=True)
        # self.conv3 = nn.Conv2d(conv2_out, conv3_out, 3)
        # self.bn3 = nn.BatchNorm2d(conv3_out, affine=True)

        # an affine operation: y = Wx + b
        self.dropout_factor = dropout_factor
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels =3 , out_channels = 8 , kernel_size = 3,stride=2),
            nn.BatchNorm2d(8, affine=True),##affine参数设为True表示weight和bias将被使用
            nn.ReLU6(inplace=True),#inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, stride=1,padding=1, groups=8),
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =8 , out_channels = 16 , kernel_size = 3,stride=1,padding=1),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, stride=1,padding=1, groups=16),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =16 , out_channels = 8 , kernel_size = 3,stride=1,padding=1),
            nn.BatchNorm2d(8, affine=True),
            )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels =8 , out_channels = 16 , kernel_size = 1,stride=1),
            nn.BatchNorm2d(16, affine=True),##affine参数设为True表示weight和bias将被使用
            nn.ReLU6(inplace=True),#inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, stride=1,padding=1, groups=16),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =16 , out_channels = 8 , kernel_size = 3,stride=1,padding=1),
            nn.BatchNorm2d(8, affine=True),
            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels =8 , out_channels = 16 , kernel_size = 1,stride=1),
            nn.BatchNorm2d(16, affine=True),##affine参数设为True表示weight和bias将被使用
            nn.ReLU6(inplace=True),#inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, stride=1,padding=1, groups=16),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =16 , out_channels = 8 , kernel_size = 3,stride=1,padding=1),
            nn.BatchNorm2d(8, affine=True),
            )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels =8 , out_channels = 16 , kernel_size = 1,stride=1),
            nn.BatchNorm2d(16, affine=True),##affine参数设为True表示weight和bias将被使用
            nn.ReLU6(inplace=True),#inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, stride=1,padding=1, groups=16),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =16 , out_channels = 8 , kernel_size = 3,stride=1,padding=1),
            nn.BatchNorm2d(8, affine=True),
            )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels =8 , out_channels = 32 , kernel_size = 1,stride=2),
            nn.BatchNorm2d(32, affine=True),##affine参数设为True表示weight和bias将被使用
            nn.ReLU6(inplace=True),#inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3, stride=1,padding=1, groups=32),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =32 , out_channels = 16 , kernel_size = 1,stride=1,padding=1),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=1, stride=1,padding=1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =32 , out_channels = 32 , kernel_size = 3,stride=1,padding=1,groups=32),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =32 , out_channels = 16 , kernel_size = 3,stride=1,padding=1),
            )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels =16 , out_channels = 32 , kernel_size = 1,stride=1),
            nn.BatchNorm2d(32, affine=True),##affine参数设为True表示weight和bias将被使用
            nn.ReLU6(inplace=True),#inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3, stride=1,padding=1, groups=32),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =32 , out_channels = 16 , kernel_size = 3,stride=1,padding=1),
            nn.BatchNorm2d(16, affine=True),
            )

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels =16 , out_channels = 64 , kernel_size = 1,stride=2),
            nn.BatchNorm2d(64, affine=True),##affine参数设为True表示weight和bias将被使用
            nn.ReLU6(inplace=True),#inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1,padding=1, groups=64),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels =64 , out_channels = 16 , kernel_size = 1,stride=1,padding=1),
            nn.BatchNorm2d(16, affine=True),
            )
        self.branch5_1 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size=3, stride=1,padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            )
        self.branch5_2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size=3, stride=1,padding=1),
            )

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, stride=2,padding=0),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64, affine=True),
            )

        self.stage5 = nn.Sequential(
            nn.Dropout(self.dropout_factor),
            nn.Linear(1600, n_class),
            )

    def forward(self, x):#
        x = self.stage1(x)
        x1 = x
        x1 = self.branch1(x1)
        x = x + x1

        x1 = x
        x1 = self.branch2(x1)
        x = x + x1

        x1 = x
        x1 = self.branch3(x1)
        x = x + x1

        x = self.stage2 (x)

        x1 = x
        x1 = self.branch4(x1)
        x = x + x1

        x = self.stage3(x)

        x1 = x
        x2 = x

        x1 = self.branch5_1(x1)
        x2 = self.branch5_2(x2)

        x = x1+x2+x

        # print(x1.size())
        # print(x2.size())

        x = self.stage4(x)
        x = x.view(-1, self.num_flat_features(x))

        # print(x.size())
        x = self.stage5(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    dummy_input = torch.randn([1, 3, 56,56])
    cfg = {}

    model = G_Net(n_class =2, input_size=56,dropout_factor=0.5)
    print(model)
    output = model(dummy_input)
    print(output.size())
