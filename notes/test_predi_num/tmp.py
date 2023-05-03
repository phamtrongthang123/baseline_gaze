import einops
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from torch.optim.lr_scheduler import StepLR
def build_loss(pred, target, mask):
    # torch.Size([max_number_sent/gt_length, 50, vocab_size]) torch.Size([1, gt_length, 50]) torch.Size([1, gt_length, 50])
    # target = einops.rearrange(target, 'b s l -> (b s) l') # torch.Size([3, 50])
    # mask = einops.rearrange(mask, 'b s l -> (b s) l') # torch.Size([3, 50])
    # one_hot = torch.nn.functional.one_hot(target, self.config['vocab_size'])
    # gt_number_sent = target.shape[0]
    # output = -(one_hot * pred[:gt_number_sent] * mask[:, :, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
    # return output.mean()
    # print(pred.shape, target.shape, mask.shape)
    # target = einops.rearrange(target, "b s l -> (b s) l")  # torch.Size([3, 50])
    # mask = einops.rearrange(mask, "b s l -> (b s) l")  # torch.Size([3, 50])
    gt_number_sent = target.shape[0]
    N, T, V = pred[:gt_number_sent].shape

    x_flat = pred[:gt_number_sent].reshape(N * T, V)
    y_flat = target.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(
        x_flat, y_flat, reduction="none", label_smoothing=0.1
    )
    loss = torch.mul(loss, mask_flat)
    loss = torch.mean(loss)
    return loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.number_prediction = nn.Sequential(nn.Linear(512, 512//2),nn.ReLU(), nn.Linear(512//2, 3))
    

    def forward(self, x):
        return self.number_prediction(x)


data = torch.randn((1,1,512))
target = torch.tensor([[1]])
mask = torch.tensor([[1.0]])
model = Net()

optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-3)
for ep in range(100):
    optimizer.zero_grad()
    num = model(data)
    print(torch.argmax(num, dim=2), num)
    loss = build_loss(num, target, mask)
    loss.backward()
    optimizer.step()
    print(loss.item())