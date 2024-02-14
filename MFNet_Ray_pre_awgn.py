# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.random.manual_seed(123) # 训练时使用这个
# torch.random.manual_seed(4321) # 合法信道测试时使用这个
torch.random.manual_seed(54321) # 窃听信道测试时使用这个

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class ConvBnLeakyRelu2d_1(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=1, groups=1):
        super(ConvBnLeakyRelu2d_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=1)


class MiniInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left  = ConvBnLeakyRelu2d(in_channels,   out_channels//2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels,   out_channels//2, padding=2, dilation=2)
        self.conv2_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
        self.conv3_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
    def forward(self,x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x


# precoding network
class Precoding(nn.Module):
    def __init__(self, nr, nt, out1, out2):
        super(Precoding, self).__init__()
        self.linear1 = nn.Linear(in_features=2 * nt * nr, out_features=out1, bias=True)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(in_features=out1, out_features=out2, bias=True)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(in_features=out2, out_features=2 * nt * nt, bias=True)
        # self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x = self.linear1(x)
        x = self.leakyrelu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.leakyrelu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        # x = self.leakyrelu3(x)
        return x


class PostProcess(nn.Module):
    def __init__(self, in_channels, nt, nr):
        super(PostProcess, self).__init__()
        self.conv1 = ConvBnLeakyRelu2d_1(in_channels=in_channels, out_channels=2*nt, kernel_size=(2, nr), padding=0, stride=(2, nr))
    def forward(self, x):
        x = self.conv1(x)
        return x


class MFNet_Ray(nn.Module):
    def __init__(self, n_class):
        super(MFNet_Ray, self).__init__()
        num_ch = [16, 48, 48, 96, 96, 48, 16, 2]
        pre_ch = [128, 128]
        self.Pmax = torch.tensor(1)
        self.noise_power = torch.tensor(1)
        self.nt = 2
        self.nr = 4
        self.K = 2
        self.K_UAV = 1
        self.out_fusionchannel = int(self.nr / self.nt)

        self.conv1_rgb   = ConvBnLeakyRelu2d(3, num_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(num_ch[0], num_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(num_ch[1], num_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(num_ch[1], num_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(num_ch[2], num_ch[2])
        self.conv4_rgb   = MiniInception(num_ch[2], num_ch[3])
        self.conv5_rgb   = MiniInception(num_ch[3], num_ch[4])
        self.conv6_rgb   = ConvBnLeakyRelu2d(num_ch[4], num_ch[5])
        self.conv7_rgb   = ConvBnLeakyRelu2d(num_ch[5], num_ch[6])
        self.conv8_rgb   = ConvBnLeakyRelu2d(num_ch[6], num_ch[7])
        self.rgb_signal_norm = nn.BatchNorm2d(1)

        self.conv1_inf   = ConvBnLeakyRelu2d(1, num_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(num_ch[0], num_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(num_ch[1], num_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(num_ch[1], num_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(num_ch[2], num_ch[2])
        self.conv4_inf   = MiniInception(num_ch[2], num_ch[3])
        self.conv5_inf   = MiniInception(num_ch[3], num_ch[4])
        self.conv6_inf   = ConvBnLeakyRelu2d(num_ch[4], num_ch[5])
        self.conv7_inf   = ConvBnLeakyRelu2d(num_ch[5], num_ch[6])
        self.conv8_inf   = ConvBnLeakyRelu2d(num_ch[6], num_ch[7])
        self.inf_signal_norm = nn.BatchNorm2d(1)

        self.precoding_rgb = Precoding(self.nr, self.nt, pre_ch[0], pre_ch[1])
        self.precoding_inf = Precoding(self.nr, self.nt, pre_ch[0], pre_ch[1])
        self.post_process = PostProcess(1, self.nt, self.nr)

        self.decode7     = ConvBnLeakyRelu2d(num_ch[7], num_ch[6])
        self.decode6     = ConvBnLeakyRelu2d(num_ch[6], num_ch[5])
        self.decode5     = ConvBnLeakyRelu2d(num_ch[5], num_ch[4])
        self.decode4     = ConvBnLeakyRelu2d(num_ch[4], num_ch[3])
        self.decode3     = ConvBnLeakyRelu2d(num_ch[3], num_ch[2])
        self.decode2     = ConvBnLeakyRelu2d(num_ch[2], num_ch[1])
        self.decode1     = ConvBnLeakyRelu2d(num_ch[1], num_ch[0])
        self.decode0     = ConvBnLeakyRelu2d(num_ch[0], n_class)
    def forward(self, x):
        # split data into RGB and INF
        x_rgb = x[:, :3]
        x_inf = x[:, 3:]

        # encode
        x_rgb    = self.conv1_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb, kernel_size=2, stride=2) # pool1
        x_rgb    = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p2, kernel_size=2, stride=2) # pool2
        x_rgb    = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p3, kernel_size=2, stride=2) # pool3
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p4, kernel_size=2, stride=2) # pool4
        x_rgb    = self.conv5_rgb(x_rgb)
        x_rgb    = self.conv6_rgb(x_rgb)
        x_rgb    = self.conv7_rgb(x_rgb)
        x_rgb    = self.conv8_rgb(x_rgb)

        x_inf    = self.conv1_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf, kernel_size=2, stride=2) # pool1
        x_inf    = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p2, kernel_size=2, stride=2) # pool2
        x_inf    = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p3, kernel_size=2, stride=2) # pool3
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p4, kernel_size=2, stride=2) # pool4
        x_inf    = self.conv5_inf(x_inf)
        x_inf    = self.conv6_inf(x_inf)
        x_inf    = self.conv7_inf(x_inf)
        x_inf    = self.conv8_inf(x_inf)
        
        # signal power normalization
        x_rgb_p5 = torch.reshape(x_rgb, [x_rgb.size()[0], 1, 1, -1])
        x_rgb_p6 = (torch.sqrt(torch.tensor(0.5)) * self.rgb_signal_norm(x_rgb_p5)).reshape([x_rgb.size()[0], 1, 2, -1])
        x_inf_p5 = torch.reshape(x_inf, [x_inf.size()[0], 1, 1, -1])
        x_inf_p6 = (torch.sqrt(torch.tensor(0.5)) * self.inf_signal_norm(x_inf_p5)).reshape([x_inf.size()[0], 1, 2, -1])

        ########################################################
        """
        Rayleigh channels
        """
        # H_rgb = (torch.sqrt(torch.tensor(0.5)) * torch.randn([x_rgb_p6.size()[0], 2, self.nr, self.nt])).to(device)  # image信道矩阵
        # H_inf = (torch.sqrt(torch.tensor(0.5)) * torch.randn([x_inf_p6.size()[0], 2, self.nr, self.nt])).to(device)  # infrared信道矩阵

        '''
        Rice channels
        '''
        H_rgb = torch.sqrt(torch.tensor(self.K / (self.K + 1))) + torch.tensor(1 / (self.K + 1)) * (
                    torch.sqrt(torch.tensor(0.5)) * torch.randn([x_rgb_p6.size()[0], 2, self.nr, self.nt])).to(device)  # image信道矩阵
        H_inf = torch.sqrt(torch.tensor(self.K_UAV / (self.K_UAV + 1))) + torch.tensor(1 / (self.K_UAV + 1)) * (
                    torch.sqrt(torch.tensor(0.5)) * torch.randn([x_rgb_p6.size()[0], 2, self.nr, self.nt])).to(device)  # infrared信道矩阵

        """
        multi-user satellite relay channels, Rice
        """
        # G = torch.sqrt(torch.tensor(self.K / (self.K + 1))) + torch.tensor(1 / (self.K + 1)) * (
        #             torch.sqrt(torch.tensor(0.5)) * torch.randn([x_rgb_p6.size()[0], 2, self.nr, self.nr])).to(device)

        # H_user1 = torch.sqrt(torch.tensor(self.K / (self.K + 1))) + torch.tensor(1 / (self.K + 1)) * (
        #             torch.sqrt(torch.tensor(0.5)) * torch.randn([x_rgb_p6.size()[0], 2, self.nr, self.nt])).to(device)  # image信道矩阵
        # H_user2 = torch.sqrt(torch.tensor(self.K / (self.K + 1))) + torch.tensor(1 / (self.K + 1)) * (
        #             torch.sqrt(torch.tensor(0.5)) * torch.randn([x_rgb_p6.size()[0], 2, self.nr, self.nt])).to(device)  # infrared信道矩阵

        # G1 = torch.complex(G[:, 0, :, :], G[:, 1, :, :]).to(device)
        # H_user1_complex = torch.complex(H_user1[:, 0, :, :], H_user1[:, 1, :, :]).to(device)
        # H_user2_complex = torch.complex(H_user2[:, 0, :, :], H_user2[:, 1, :, :]).to(device)

        # H_rgb_1 = torch.matmul(G1, H_user1_complex).to(device)
        # H_inf_1 = torch.matmul(G1, H_user2_complex).to(device)

        # H_rgb_real = torch.unsqueeze(H_rgb_1.real, dim=1).to(device)
        # H_rgb_imag = torch.unsqueeze(H_rgb_1.imag, dim=1).to(device)

        # H_inf_real = torch.unsqueeze(H_inf_1.real, dim=1).to(device)
        # H_inf_imag = torch.unsqueeze(H_inf_1.imag, dim=1).to(device)

        # H_rgb = torch.cat([H_rgb_real, H_rgb_imag], dim=1).to(device)
        # H_inf = torch.cat([H_inf_real, H_inf_imag], dim=1).to(device)


        ########################################################

        h_rgb = torch.reshape(H_rgb, [H_rgb.size()[0], 1, 1, -1]).to(device)
        h_inf = torch.reshape(H_inf, [H_inf.size()[0], 1, 1, -1]).to(device)

        # precoding
        F_rgb = self.precoding_rgb(h_rgb).reshape([h_rgb.size()[0], 2, self.nt, self.nt])
        F_inf = self.precoding_inf(h_inf).reshape([h_inf.size()[0], 2, self.nt, self.nt])

        # precoding power normalization
        z_rgb_1 = torch.zeros([F_rgb.size()[0], 2, self.nt, self.nt]).to(device)
        for i in range(F_rgb.size()[0]):
            z_rgb_1[i, ::] = F_rgb[i, ::] * torch.sqrt(self.Pmax / (torch.matmul(F_rgb[i, 0, ::], F_rgb[i, 0, ::].T)
                            + torch.matmul(F_rgb[i, 1, ::], F_rgb[i, 1, ::].T)).trace())

        z_inf_1 = torch.zeros([F_inf.size()[0], 2, self.nt, self.nt]).to(device)
        for i in range(F_inf.size()[0]):
            z_inf_1[i, ::] = F_inf[i, ::] * torch.sqrt(self.Pmax / (torch.matmul(F_inf[i, 0, ::], F_inf[i, 0, ::].T)
                             + torch.matmul(F_inf[i, 1, ::], F_inf[i, 1, ::].T)).trace())

        '''
        rgb image: through precoding and channel
        '''
        # CNN precoding layer and weights
        z_rgb_2 = torch.cat(torch.split(z_rgb_1, 1, dim=0), dim=1)
        pre_weight_rgb = torch.zeros([F_rgb.size()[0] * 2 * self.nt, 1, 2, self.nt]).to(device)
        for i in range(F_rgb.size()[0]):
            for t in range(2 * self.nt):
                if t % 2 == 0:
                    pre_weight_rgb[2 * self.nt * i + t, 0, ::] = torch.cat(
                        [torch.unsqueeze(z_rgb_2[0, 2 * i, int(t / 2), :], dim=0),
                         torch.unsqueeze(-z_rgb_2[0, 2 * i + 1, int(t / 2), :], dim=0)], dim=0)
                else:
                    pre_weight_rgb[2 * self.nt * i + t, 0, ::] = torch.cat(
                        [torch.unsqueeze(z_rgb_2[0, 2 * i + 1, int((t - 1) / 2), :], dim=0),
                         torch.unsqueeze(z_rgb_2[0, 2 * i, int((t - 1) / 2), :], dim=0)], dim=0)
        Precoding_rgb = nn.Conv2d(in_channels=F_rgb.size()[0], out_channels=F_rgb.size()[0] * 2 * self.nt,
                                  kernel_size=(2, self.nt), stride=(2, self.nt), bias=False, groups=F_rgb.size()[0])
        Precoding_rgb.weight = nn.Parameter(data=pre_weight_rgb, requires_grad=False)

        # CNN channel layer and weights
        H_rgb_2 = torch.cat(torch.split(H_rgb, 1, dim=0), dim=1)
        channel_weight_rgb = torch.zeros([H_rgb.size()[0] * 2 * self.nr, 1, 2, self.nt]).to(device)
        for i in range(H_rgb.size()[0]):
            for t in range(2 * self.nr):
                if t % 2 == 0:
                    channel_weight_rgb[2 * self.nr * i + t, 0, ::] = torch.cat(
                        [torch.unsqueeze(H_rgb_2[0, 2 * i, int(t / 2), :], dim=0),
                         torch.unsqueeze(-H_rgb_2[0, 2 * i + 1, int(t / 2), :], dim=0)], dim=0)
                else:
                    channel_weight_rgb[2 * self.nr * i + t, 0, ::] = torch.cat(
                        [torch.unsqueeze(H_rgb_2[0, 2 * i + 1, int((t - 1) / 2), :], dim=0),
                         torch.unsqueeze(H_rgb_2[0, 2 * i, int((t - 1) / 2), :], dim=0)], dim=0)
        Channel_rgb = nn.Conv2d(in_channels=H_rgb.size()[0], out_channels=H_rgb.size()[0] * 2 * self.nr,
                                kernel_size=(2, self.nt), stride=(2, self.nt), bias=False, groups=H_rgb.size()[0])
        Channel_rgb.weight = nn.Parameter(data=channel_weight_rgb, requires_grad=False)

        # precoding and dimentional transform
        input_rgb_ = torch.cat(torch.split(x_rgb_p6, 1, dim=0), dim=1)
        output_rgb = Precoding_rgb(input_rgb_)
        output_rgb_ = torch.cat(torch.split(output_rgb, 2 * self.nt, dim=1), dim=0)
        x_rgb_p6_oven = output_rgb_[:, 0::2, ::].clone().transpose(1, 3)  # 偶数行
        x_rgb_p6_odd = output_rgb_[:, 1::2, ::].clone().transpose(1, 3)  # 奇数行
        x_rgb_p7_oven = torch.reshape(x_rgb_p6_oven, [x_rgb_p6_oven.size()[0], 1, 1, -1])
        x_rgb_p7_odd = torch.reshape(x_rgb_p6_odd, [x_rgb_p6_odd.size()[0], 1, 1, -1])
        x_rgb_hat = torch.cat([x_rgb_p7_oven, x_rgb_p7_odd], dim=2)

        # ----------------through channel-------------------
        input_rgb1_ = torch.cat(torch.split(x_rgb_hat, 1, dim=0), dim=1)
        output_rgb1 = Channel_rgb(input_rgb1_)
        output_rgb1_ = torch.cat(torch.split(output_rgb1, 2 * self.nr, dim=1), dim=0)
        x_rgb_p6_oven1 = output_rgb1_[:, 0::2, ::].clone().transpose(1, 3)  # 偶数行
        x_rgb_p6_odd1 = output_rgb1_[:, 1::2, ::].clone().transpose(1, 3)  # 奇数行
        x_rgb_p7_oven1 = torch.reshape(x_rgb_p6_oven1, [x_rgb_p6_oven1.size()[0], 1, 1, -1])
        x_rgb_p7_odd1 = torch.reshape(x_rgb_p6_odd1, [x_rgb_p6_odd1.size()[0], 1, 1, -1])
        y_rgb_hat = torch.cat([x_rgb_p7_oven1, x_rgb_p7_odd1], dim=2)
        Y_rgb = torch.reshape(y_rgb_hat, [x_rgb.size()[0], self.out_fusionchannel * x_rgb.size()[1], x_rgb.size()[2], x_rgb.size()[3]])

        '''
        infrared image: through precoding and channel
        '''
        # CNN precoding layer and weights
        z_inf_2 = torch.cat(torch.split(z_inf_1, 1, dim=0), dim=1)
        pre_weight_inf = torch.zeros([F_inf.size()[0] * 2 * self.nt, 1, 2, self.nt]).to(device)
        for i in range(F_inf.size()[0]):
            for t in range(2 * self.nt):
                if t % 2 == 0:
                    pre_weight_inf[2 * self.nt * i + t, 0, ::] = torch.cat(
                        [torch.unsqueeze(z_inf_2[0, 2 * i, int(t / 2), :], dim=0),
                         torch.unsqueeze(-z_inf_2[0, 2 * i + 1, int(t / 2), :], dim=0)], dim=0)
                else:
                    pre_weight_inf[2 * self.nt * i + t, 0, ::] = torch.cat(
                        [torch.unsqueeze(z_inf_2[0, 2 * i + 1, int((t - 1) / 2), :], dim=0),
                         torch.unsqueeze(z_inf_2[0, 2 * i, int((t - 1) / 2), :], dim=0)], dim=0)
        Precoding_inf = nn.Conv2d(in_channels=F_inf.size()[0], out_channels=F_inf.size()[0] * 2 * self.nt,
                                  kernel_size=(2, self.nt), stride=(2, self.nt), bias=False, groups=F_inf.size()[0])
        Precoding_inf.weight = nn.Parameter(data=pre_weight_inf, requires_grad=False)

        # CNN channel layer and weights
        H_inf_2 = torch.cat(torch.split(H_inf, 1, dim=0), dim=1)
        channel_weight_inf = torch.zeros([H_inf.size()[0] * 2 * self.nr, 1, 2, self.nt]).to(device)
        for i in range(H_inf.size()[0]):
            for t in range(2 * self.nr):
                if t % 2 == 0:
                    channel_weight_inf[2 * self.nr * i + t, 0, ::] = torch.cat(
                        [torch.unsqueeze(H_inf_2[0, 2 * i, int(t / 2), :], dim=0),
                         torch.unsqueeze(-H_inf_2[0, 2 * i + 1, int(t / 2), :], dim=0)], dim=0)
                else:
                    channel_weight_inf[2 * self.nr * i + t, 0, ::] = torch.cat(
                        [torch.unsqueeze(H_inf_2[0, 2 * i + 1, int((t - 1) / 2), :], dim=0),
                         torch.unsqueeze(H_inf_2[0, 2 * i, int((t - 1) / 2), :], dim=0)], dim=0)
        Channel_inf = nn.Conv2d(in_channels=H_inf.size()[0], out_channels=H_inf.size()[0] * 2 * self.nr,
                                kernel_size=(2, self.nt), stride=(2, self.nt), bias=False, groups=H_inf.size()[0])
        Channel_inf.weight = nn.Parameter(data=channel_weight_inf, requires_grad=False)

        # precoding and dimentional transform
        input_inf_ = torch.cat(torch.split(x_inf_p6, 1, dim=0), dim=1)
        output_inf = Precoding_inf(input_inf_)
        output_inf_ = torch.cat(torch.split(output_inf, 2 * self.nt, dim=1), dim=0)
        x_inf_p6_oven = output_inf_[:, 0::2, ::].clone().transpose(1, 3)  # 偶数行
        x_inf_p6_odd = output_inf_[:, 1::2, ::].clone().transpose(1, 3)  # 奇数行
        x_inf_p7_oven = torch.reshape(x_inf_p6_oven, [x_inf_p6_oven.size()[0], 1, 1, -1])
        x_inf_p7_odd = torch.reshape(x_inf_p6_odd, [x_inf_p6_odd.size()[0], 1, 1, -1])
        x_inf_hat = torch.cat([x_inf_p7_oven, x_inf_p7_odd], dim=2)

        # through channel
        input_inf1_ = torch.cat(torch.split(x_inf_hat, 1, dim=0), dim=1)
        output_inf1 = Channel_inf(input_inf1_)
        output_inf1_ = torch.cat(torch.split(output_inf1, 2 * self.nr, dim=1), dim=0)
        x_inf_p6_oven1 = output_inf1_[:, 0::2, ::].clone().transpose(1, 3)  # 偶数行
        x_inf_p6_odd1 = output_inf1_[:, 1::2, ::].clone().transpose(1, 3)  # 奇数行
        x_inf_p7_oven1 = torch.reshape(x_inf_p6_oven1, [x_inf_p6_oven1.size()[0], 1, 1, -1])
        x_inf_p7_odd1 = torch.reshape(x_inf_p6_odd1, [x_inf_p6_odd1.size()[0], 1, 1, -1])
        y_inf_hat = torch.cat([x_inf_p7_oven1, x_inf_p7_odd1], dim=2)
        # Y_inf = torch.reshape(y_inf_hat, [x_inf.size()[0], self.out_fusionchannel * x_inf.size()[1], x_inf.size()[2], x_inf.size()[3]])

        noise = torch.randn([y_inf_hat.size()[0], y_inf_hat.size()[1], y_inf_hat.size()[2], y_inf_hat.size()[3]])
        awgn_noise = (torch.sqrt(self.noise_power * torch.tensor(0.5)) * noise).to(device)
        x_fusion_1 = torch.add(y_rgb_hat, y_inf_hat)
        x_fusion_1 = torch.add(x_fusion_1, awgn_noise)
        x_fusion = self.post_process(x_fusion_1)
        x_fusion_oven1 = x_fusion[:, 0::2, ::].clone().transpose(1, 3)  # 偶数行
        x_fusion_odd1 = x_fusion[:, 1::2, ::].clone().transpose(1, 3)  # 奇数行
        x_fusion_oven2 = torch.reshape(x_fusion_oven1, [x_fusion_oven1.size()[0], 1, 1, -1])
        x_fusion_odd2 = torch.reshape(x_fusion_odd1, [x_fusion_odd1.size()[0], 1, 1, -1])
        x = torch.cat([x_fusion_oven2, x_fusion_odd2], dim=2).reshape([x_rgb.size()[0], x_rgb.size()[1], x_rgb.size()[2], x_rgb.size()[3]])
        # x = torch.add(x, awgn_noise)
        
        # decode
        x = self.decode7(x)
        x = self.decode6(x)
        x = self.decode5(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool4
        x = self.decode4(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool3
        x = self.decode3(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool2
        x = self.decode2(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool1
        x = self.decode1(x)
        x = self.decode0(x)
        return x


def unit_test():
    import numpy as np
    batch_size = 2

    model = MFNet_Ray(n_class=11)
    x = torch.tensor(np.random.rand(batch_size, 4, 480, 640).astype(np.float32)).to(device)
    # H_rgb = (torch.sqrt(torch.tensor(0.5)) * torch.randn([batch_size, 2, model.nr, model.nt])).to(device)  # image信道矩阵
    # H_ir = (torch.sqrt(torch.tensor(0.5)) * torch.randn([batch_size, 2, model.nr, model.nt])).to(device)  # infrared信道矩阵

    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (batch_size, 11, 480, 640), 'output shape (batch_size, 11, 480, 640) is expected!'
    print('test ok!')


if __name__ == '__main__':
    unit_test()
