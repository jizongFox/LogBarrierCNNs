import numpy as np

from torch.autograd import Variable

from visdom import Visdom

class Dashboard:

    def __init__(self, port=8097,env = 'LogBarrer'):
        self.vis = Visdom(port=port,server='http://turing.livia.etsmtl.ca',env=env)

    def loss(self, losses, title):
        x = np.arange(1, len(losses)+1, 1)

        self.vis.line(losses, x, env='loss', opts=dict(title=title))

    def image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy()

        self.vis.image(image.astype(np.float), env=self.vis.env, opts=dict(title=title))