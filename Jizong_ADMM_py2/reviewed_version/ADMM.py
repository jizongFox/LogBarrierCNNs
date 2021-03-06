import torch, numpy as np, torch.nn as nn, torch.nn.functional as F
from network import UNet
from criterion import CrossEntropyLoss2d
import maxflow, matplotlib.pyplot as plt


class networks(object):
    '''
    neural_net : neural network
    limage and uimage: b,c,h,w of torch.float
    gamma and s: numpy of shape b h w
    u,v: numpy of shape b h w
    '''

    def __init__(self, neural_network, lowerbound, upperbound):
        super(networks, self).__init__()
        self.lowbound = lowerbound
        self.upbound = upperbound
        self.neural_net = neural_network
        self.reset()
        self.optimiser = torch.optim.Adam(self.neural_net.parameters(), lr=0.001, weight_decay=1e-5)
        self.CEloss_criterion = CrossEntropyLoss2d()
        self.u_r = 1
        self.u_s = 1

    def limage_forward(self, limage, lmask):
        self.limage = limage
        self.lmask = lmask
        self.limage_output = self.neural_net(limage)

    def uimage_forward(self, uimage):
        self.uimage = uimage
        self.uimage_output = self.neural_net(uimage)
        if self.gamma is None:
            self.__initialize_dummy_variables(self.uimage_output)

    def heatmap2segmentation(self, heatmap):
        return heatmap.max(1)[1]

    def __initialize_dummy_variables(self, uimage_heatmap):
        self.gamma = self.heatmap2segmentation(uimage_heatmap).cpu().data.numpy()  # b, w, h
        self.s = self.gamma  # b w h
        self.u = np.zeros(list(self.gamma.shape))  # b w h
        self.v = np.zeros(self.u.shape)

    def reset(self):
        self.limage = None
        self.uimage = None
        self.lmask = None
        self.limage_output = None
        self.uimage_output = None
        self.gamma = None
        self.s = None
        self.u = None
        self.v = None

    def update_theta(self):
        for i in xrange(10):
            CE_loss = self.CEloss_criterion(self.limage_output, self.lmask.squeeze(1))
            unlabled_loss = self.u_r / 2 * (
                        self.uimage_output - torch.from_numpy(self.gamma).float().cuda() + torch.Tensor(
                    self.u).float().cuda()).norm(p=2) ** 2 + self.u_s / 2 * (
                                        self.uimage_output - torch.from_numpy(self.s).float().cuda() + torch.Tensor(
                                    self.v).float().cuda()).norm(p=2) ** 2

            loss = CE_loss + unlabled_loss
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            self.limage_forward(self.limage, self.lmask)  # shape b,c,w,h
            self.uimage_forward(self.uimage)

    def update_gamma(self):
        unary_term_gamma_1 = np.multiply((0.5 - (self.uimage_output.cpu().data.numpy()[:, 1, :, :] + self.u)),
                                         self.gamma)

        plt.figure(5)
        plt.imshow(unary_term_gamma_1.squeeze())
        plt.title('unary_term')
        plt.show(block=False)
        plt.pause(0.001)
        # unary_term_gamma_0 = -unary_term_gamma_1
        unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)
        neighbor_term = 1
        new_gamma = np.zeros(self.gamma.shape)
        g = maxflow.Graph[float](0, 0)
        i = 0
        # Add the nodes.
        nodeids = g.add_grid_nodes(list(self.gamma.shape)[1:])
        # Add edges with the same capacities.
        g.add_grid_edges(nodeids, neighbor_term)
        # Add the terminal edges.
        g.add_grid_tedges(nodeids,unary_term_gamma_1[i].astype(np.int32).squeeze(),
                          unary_term_gamma_0[i].squeeze())
        g.maxflow()
        # Get the segments.
        sgm = g.get_grid_segments(nodeids) * 1

        # The labels should be 1 where sgm is False and 0 otherwise.
        new_gamma[i] = np.int_(np.logical_not(sgm))
        g.reset()
        self.gamma = new_gamma

    def update_s(self):
        s_new = np.zeros(self.s.shape)

        a = self.uimage_output.cpu().data.numpy()[:, 1, :, :] + self.v - 0.5
        for i in range(a.shape[0]):
            a_ = a[i]
            A_plus = (a_ >= 0).sum()
            A_sub = (a_ < 0).sum()
            if A_plus >= self.upbound:
                threshold = np.sort(a_.reshape(-1))[self.upbound:][0]
                s_new[i][np.where(a_ > threshold)] = 1
            elif A_plus <= self.upbound:
                threshold = np.sort(a_.reshape(-1))[::-1][self.lowbound]
                s_new[i][np.where(a_ > threshold)] = 1

        assert self.s.shape == s_new.shape
        self.s = s_new

    def update_u(self):
        new_u = self.u + F.softmax(self.uimage_output,dim=1)[:, 1, :, :].cpu().data.numpy() - self.gamma
        assert new_u.shape == self.u.shape
        self.u = new_u

    def update_v(self):
        new_v = self.v + F.softmax(self.uimage_output,dim=1)[:, 1, :, :].cpu().data.numpy() - self.s
        assert new_v.shape == self.v.shape
        self.v = new_v

    def update(self, (limage, lmask), uimage):
        self.limage_forward(limage, lmask)
        self.uimage_forward(uimage)
        self.update_theta()
        self.update_gamma()
        self.update_s()
        self.update_u()
        self.update_v()

    def show_labeled_pair(self):
        fig = plt.figure(1,figsize=(8,8))
        fig.suptitle("labeled data", fontsize=16)

        ax1 = fig.add_subplot(221)
        ax1.imshow(self.limage[0].cpu().data.numpy().squeeze())
        ax1.title.set_text('original image')

        ax2 = fig.add_subplot(222)
        ax2.imshow(self.lmask[0].cpu().data.numpy().squeeze())
        ax2.title.set_text('ground truth')

        ax3 = fig.add_subplot(223)
        ax3.imshow(F.softmax(self.limage_output, dim=1)[0][1].cpu().data.numpy())
        ax3.title.set_text('prediction of the probability')


        ax4 = fig.add_subplot(224)
        ax4.imshow(np.abs(self.lmask[0].cpu().data.numpy().squeeze()-F.softmax(self.limage_output, dim=1)[0][1].cpu().data.numpy()))
        ax4.title.set_text('difference')


        plt.show(block=False)
        plt.pause(0.01)

    def show_ublabel_image(self):
        fig = plt.figure(2,figsize=(8,4))
        fig.suptitle("Unlabeled data", fontsize=16)


        ax1 = fig.add_subplot(121)
        ax1.imshow(self.uimage[0].cpu().data.numpy().squeeze())
        ax1.title.set_text('original image')


        ax2 = fig.add_subplot(122)
        ax2.imshow(F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy())
        ax2.title.set_text('probability prediction')

        plt.show(block=False)
        plt.pause(0.01)

    def show_gamma(self):
        plt.figure(3)
        plt.subplot(1, 1, 1)
        plt.imshow(self.gamma[0])
        plt.title('Gamma')
        plt.show(block=False)
        plt.pause(0.01)

    def show_s(self):
        plt.figure(4)
        plt.subplot(1, 1, 1)
        plt.imshow(self.s[0])
        plt.show(block=False)
        plt.pause(0.01)


if __name__ == "__main__":
    net = UNet(num_classes=2)
    net_ = networks(net, 10, 100)
    for i in xrange(10):
        # print(net_)
        limage = torch.randn(1, 1, 256, 256)
        uimage = torch.randn(1, 1, 256, 256)
        lmask = torch.randint(0, 2, (1, 256, 256), dtype=torch.long)
        net_.update((limage, lmask), uimage)
