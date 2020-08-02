import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from torchvision.transforms import transforms


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class G(nn.Module):
    def __init__(self, rvs=16):  # rdms = random vector size
        super(G, self).__init__()
        self.rvs = rvs

        self.cov = nn.Sequential(
            # (rvs, 1, 1) -> (128, 4, 4)
            nn.ConvTranspose2d(in_channels=rvs, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # (128, 4, 4) -> (256, 8, 8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (64, 16, 16) -> (3, 32, 32)
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.weight_init(0., 0.2)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.cov(x)
        return x


class D(nn.Module):
    def __init__(self, insp=28, inch=1, outf=128):  # insp = input shape, inch = input_channel, outf = output_feature
        super(D, self).__init__()
        self.insp = insp
        self.inch = inch
        self.outf = outf
        self.outs = insp - 12

        self.cov = nn.Sequential(
            # 32 -> 28
            nn.Conv2d(in_channels=inch, out_channels=16, kernel_size=5, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 28 -> 14
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 14 -> 12
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 12 -> 6
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 6 -> 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 -> 1
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()

            # nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),
            # (512, 16, 16) -> (128, 16, 16)
            # nn.Conv2d(in_channels=512, out_channels=outf, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True)
        )
        '''
        self.fc = nn.Sequential(
            nn.Linear(self.outf * self.outs * self.outs, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        '''
        self.weight_init(0., 0.2)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.cov(x)
        # print(x.shape)
        # x = x.view(-1, self.outf * self.outs * self.outs)
        # x = self.fc(x)
        return x.view(-1, 1).squeeze(1)


class GAN:
    def __init__(self, args):
        self.args = args
        self.rvs = args.rvs
        self.insp = args.insp
        self.inch = args.inch
        self.outf = args.outf

        self.transformer = transforms.RandomChoice([
            transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=0.1), transforms.ToTensor()]),
            transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(contrast=0.1), transforms.ToTensor()]),
            transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(saturation=0.1), transforms.ToTensor()]),
            transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(degrees=10, resample=False, expand=False, center=None), transforms.ToTensor()]),
            transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()]),
        ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.D = D(insp=args.insp, inch=args.inch, outf=args.outf).to(self.device)
        self.G = G(rvs=args.rvs).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.beta, 0.999))
        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.beta, 0.999))

        self.test_noise = torch.randn(args.grid_w * args.grid_h, args.rvs, 1, 1).to(device=self.device)
        self.real_label = 1
        self.fake_label = 0

    def save(self):
        torch.save(self.G.state_dict(), os.path.join('./log', '{}_{}_G.ckpt'.format(self.args.model, self.args.epochs)))
        torch.save(self.D.state_dict(), os.path.join('./log', '{}_{}_D.ckpt'.format(self.args.model, self.args.epochs)))

    def train_batch(self, epoch, idx, x, y, len):
        x, y = x.to(self.device), y.to(self.device)
        # train D with real
        self.D.zero_grad()
        minibs = x.size(0)
        label = torch.full((minibs,), self.real_label, device=self.device)
        y = self.D(x)
        loss_real = self.criterion(y, label)
        loss_real.backward()
        D_real_avg = y.mean().item()

        # train D with fake
        noise = torch.randn(minibs, self.args.rvs, 1, 1).to(self.device)
        label.fill_(self.fake_label)
        fake_y = self.G(noise)
        # print(fake_y.shape)
        y = self.D(fake_y.detach())
        # print(y.shape)
        loss_fake = self.criterion(y, label)
        loss_fake.backward()
        D_fake_avg = y.mean().item()
        loss_D = loss_real + loss_fake
        self.optimizerD.step()

        # train G
        self.G.zero_grad()
        label.fill_(self.real_label)
        y = self.D(fake_y)
        loss_G = self.criterion(y, label)
        loss_G.backward()
        G_avg = y.mean().item()
        self.optimizerG.step()

        # print log
        if idx % 100 == 0:
            print('Epoch [{}/{}], [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}, Loss_D(x): '
                  '{:.4f}, Loss_D(G(z)): {:.4f}/{:.4f}'.format(epoch, self.args.epochs, idx, len,
                                                               loss_D.item(), loss_G.item(), D_real_avg, G_avg,
                                                               D_fake_avg))
            get_grid(imgs=self.G(self.test_noise).detach().cpu().numpy(), args=self.args, epoch=epoch, it=idx)

    def train(self, dataloader):
        for epoch in range(self.args.epochs):
            for idx, (x, y) in enumerate(dataloader):
                tt = 1
                self.train_batch(epoch, idx*tt, x, y, len(dataloader)*tt)
                #for i in range(1, tt):
                #    x_ = torch.Tensor([self.transformer(i).numpy() for i in x])
                #    self.train_batch(epoch, idx*tt+i, x_, y, len(dataloader)*tt)
        self.save()

    def test(self, dataloader):
        pass

    def simple(self):
        pass


from options import get_args
from util import get_grid

if __name__ == '__main__':
    args = get_args()
    model = GAN(args=args)
    noise = torch.randn(args.bs, args.rvs, 1, 1).to(model.device)
    gg = model.G(noise)
    print(gg.shape)
    get_grid(imgs=gg.detach().cpu().numpy(), args=args, epoch=1, it=1)
    # print(model.D(gg))
