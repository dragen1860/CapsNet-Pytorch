import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms 
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter

# change batch size for your GPU card !!!
glo_batch_size = 180

data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))
                ])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', train = True, download = True, transform = data_transform),
                    batch_size = glo_batch_size, shuffle = True
                    )
test_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', train = False, download = True, transform = data_transform),
                    batch_size = glo_batch_size, shuffle = True
                    )


class CapsNet(nn.Module):
    conv1_kernel_size = 9
    conv1_kernel_num = 256
    conv1_stride = 1

    caps1_conv_kernel_size = 9
    caps1_conv_kernel_num = 32
    caps1_conv1_stride = 2
    caps1_num = 8

    output_num = 10
    output_size = 16

    batch_size = glo_batch_size

    def __init__(self, dataloader):
        super(CapsNet, self).__init__()
        self.dataloader = dataloader

        self.build()

    def create_cell_fn(self):
        """
        create sub-network inside a capsule.
        :return:
        """
        conv1 = nn.Conv2d(self.conv1_kernel_num, self.caps1_conv_kernel_num, kernel_size = self.caps1_conv_kernel_size, stride = self.caps1_conv1_stride)
        #relu = nn.ReLU(inplace = True)
        #net = nn.Sequential(conv1, relu)
        return conv1


    def build(self):
        conv1 = nn.Conv2d(1, self.conv1_kernel_num, kernel_size = self.conv1_kernel_size, stride = self.conv1_stride)
        relu1 = nn.ReLU(inplace = True)

        caps1_cells = [self.create_cell_fn() for i in range(self.caps1_num)]
        cap1 = Caps(caps1_cells)

        route1 = Route(self.caps1_num, 6*6*self.caps1_conv_kernel_num, self.output_num, self.output_size, batch_size = self.batch_size)

        self.net = nn.Sequential(conv1, relu1, cap1, route1)


    def forward(self, input):
        return self.net(input)


    def margin_loss(self, input, target):
        v_mod = torch.sqrt(torch.mul(input,input).sum(dim = 2, keepdim = True))

        m_plus = 0.9
        m_minus = 0.1
        zero_val = Variable(torch.zeros(1)).cuda()
        max_l = torch.max(m_plus - v_mod, zero_val).view(self.batch_size, -1)
        max_r = torch.max(v_mod - m_minus, zero_val).view(self.batch_size, -1)
        Lc = target * max_l + 0.5 * (1 - target) * (max_r)
        Lc = Lc.sum(dim = 1).mean()

        return Lc




class Route(nn.Module):

    def __init__(self, in_caps_num, in_caps_size, out_caps_num, out_caps_size, batch_size):
        super(Route, self).__init__()
        self.in_caps_num = in_caps_num
        self.in_caps_size = in_caps_size
        self.out_caps_num = out_caps_num
        self.out_caps_size = out_caps_size
        self.batch_size = glo_batch_size

        #we can not use batch_size for W parameters as the network does not include batch factor.
        self.W = nn.Parameter(torch.randn(1, in_caps_size, out_caps_num, out_caps_size, in_caps_num))

    def softmax(self, input, dim):
        input_ex = torch.exp(input)
        return input_ex / input_ex.sum(dim, keepdim = True)

    def squash(self, input):
        mod_sq = torch.sum(input**2, dim = 2, keepdim = True)
        mod = torch.sqrt(mod_sq)
        return (mod / (1 + mod)) * (input / mod_sq)

    def forward(self, input):
        #input (batch, in_caps_num, in_caps_size) => (batch, in_caps_size, in_caps_num)
        input = torch.transpose(input, 1, 2)
        #input (batch, vectorsin_caps_size, in_caps_num) =.(batch, in_caps_size, out_caps_num, in_caps_num, 1)
        input = torch.stack([input]*self.out_caps_num, dim = 2).unsqueeze(4)
        #u_hat     : (batch, in_caps_size, out_caps_num, out_caps_size,     1) 
        #          = (batch, in_caps_size, out_caps_num, out_caps_size, in_caps_num)
        #          * (batch, in_caps_size, out_caps_num, in_caps_num,       1)
        u_hat = torch.matmul(torch.cat([self.W] * self.batch_size, 0) , input)
        #initialzie b_ij according to prior prob distribution
        #b_ij (1, in_caps_size, out_caps_num, 1), do not include batch_size
        b_ij = Variable(torch.zeros(1, self.in_caps_size, self.out_caps_num, 1)).cuda()

        #start routing now.
        for _ in range(3):
            #convert to coupling parameters, (batch_size, in_caps_size, out_caps_num, 1) 
            c_ij = self.softmax(b_ij, dim = 2)
            c_ij = torch.cat([c_ij] * self.batch_size, dim = 0).unsqueeze(4)

            #Here using broadcasting mechnism.
            #s_j : (batch, in_caps_size, out_caps_num, out_caps_size, 1)
            #    = (batch, in_caps_size, out_caps_num, 1,             1)
            #    * (batch, in_caps_size, out_caps_num, out_caps_size, 1)
            #sum:  (batch,      1      , out_caps_num, out_caps_size, 1)
            s_j = torch.mul(c_ij, u_hat).sum(dim = 1, keepdim = True)
            #squash s_j to v_j (batch, 1, out_caps_num, out_caps_size, 1)
            v_j = self.squash(s_j)
            #in order to satifiy u_hat * v_j, we expand its dimension
            #=> (batch, in_caps_size, out_caps_num, out_caps_size, 1)
            v_j_ext = torch.cat([v_j] * self.in_caps_size, dim = 1)
            #u_hat transpose => (batch, in_caps_size, out_caps_num, 1, out_caps_size) 
            #v_j_ext            (batch, in_caps_size, out_caps_num, out_caps_size, 1)
            #matmul          => (batch, in_caps_size, out_caps_num, 1   , 1)
            #seueeze         => (batch, in_caps_size, out_caps_num, 1)
            #mean            => (  1  , in_caps_size, out_caps_num, 1)
            uv = torch.matmul(u_hat.transpose(3,4), v_j_ext).squeeze(4).mean(dim = 0, keepdim = True)
            #update b_ij
            b_ij = b_ij + uv


        #return (batch, out_caps_num, out_caps_size, 1)
        return v_j.squeeze(1)






class Caps(nn.Module):
    """
    Capsule layer, this layer is a wrapper of any kinds of sub-layer inside single capsule. When initialized, it received a create_cell_fn to create each 
    sub network for each capsules and compute each capsule output.
    In the feature, we can put all current network as a capsule unit.
    """
    def __init__(self, cells): 
        super(Caps, self).__init__()
        self.cells = cells 
        self.caps_num = len(cells)

        for i, cell in enumerate(cells):
            self.add_module("subnet"+str(i),cell)

    def forward(self, input):
        #u=[val] : val: (batch, channels, height, width)
        u = [self.cells[i](input) for i in range(self.caps_num)]
        # stack the caps_num axis before channels axis.
        #=> (batch, caps_num, channels, height, width)
        u = torch.stack(u, dim = 1)
        #flat to (batch, caps_num, output)
        u = u.view(input.size(0), self.caps_num, -1)
        #squash output
        return self.squash(u)

    def squash(self, input):
        mod_sq = torch.sum(input**2, dim = 2, keepdim = True)
        mod = torch.sqrt(mod_sq)
        return (mod / (1 + mod)) * (input / mod_sq)


def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot


if __name__ == '__main__':
    net = CapsNet(train_loader)
    net.cuda()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr = 5e-4)
    tb = SummaryWriter()

    best_accuracy = 0

    if os.path.exists('caps.mdl'):
        with open('caps.mdl','rb') as f:
            net = torch.load('caps.mdl')
            print('loaded mdl yet.')

    for epoch in range(300):

        net.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, total = len(train_loader), ncols=100, leave=False, unit='b'+str(epoch))):
            target_onehot = to_one_hot(target, 10)
            data, target = Variable(data).cuda(), Variable(target_onehot).cuda()

            optimizer.zero_grad()
            output = net(data)
            loss = net.margin_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0 and batch_idx != 0:
                tb.add_scalar('loss', loss.data[0])

        net.eval()
        correct_prediction = 0.0
        total_counter = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = Variable(data).cuda(), Variable(target.type(torch.LongTensor)).cuda()

            #pred [batch, out_caps_num, out_caps_size, 1]
            pred = net(data)
            # pred_mod [batch, out_caps_num, 1, 1] => [batch, out_caps_num]
            pred_mod = pred.mul(pred).sum(dim = 2).sqrt().squeeze(2)
            # v1 [batch]
            _ , v1 = torch.max(pred_mod , dim = 1)
            correct_prediction += target.eq(v1).sum().data.cpu().numpy()[0]
            total_counter += glo_batch_size

            if batch_idx % 1000 == 0:
                tb.add_scalar('accuracy', correct_prediction/(total_counter))
                break

        print(epoch, 'test accuracy:', correct_prediction/(total_counter))
        if best_accuracy < correct_prediction/(total_counter) :
            best_accuracy = correct_prediction/(total_counter) 
            torch.save(net, 'caps.mdl')
            print('saved to mdl file.')

    tb.close()





