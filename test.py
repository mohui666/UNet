import tifffile
import torch

from model import u_net

net = u_net()
Y = tifffile.imread("ISBI-2012-challenge/test-volume.tif")[0]
y = torch.from_numpy(Y)
y = y.to(dtype=torch.float32)
y = y.unsqueeze(0).unsqueeze(0)
print(net(y).shape)
