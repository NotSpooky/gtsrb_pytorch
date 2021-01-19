import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np

# Assumes a gtsrb folder inside next to this file

BATCH_SIZE = 32
EPOCHS = 10
TEST_SIZE = 0.4
NUM_CATEGORIES = 43

def to_categorical (y):
    to_ret = numpy.zeros (
        (y.shape [0], numpy.unique (y).size), dtype=numpy.uint8
    )
    to_ret [
        numpy.arange (y.shape [0]), numpy.uint8 (y)
    ] = 1
    return temp_outs

class Net (nn.Module):

    def __init__ (self):
        super (Net, self).__init__()
        # 3 input image channels, 16 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d (3, 16 * 3, 3, groups = 3)
        self.conv2 = nn.Conv2d (16 * 3, 8 * 3, 3, groups = 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear (8 * 3 * 6 * 6, 32)  # 6*6 from image dimension
        self.fc2 = nn.Linear (32, 32)
        self.fc3 = nn.Linear (32, NUM_CATEGORIES)

    def forward (self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d (F.selu (self.conv1 (x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d (F.selu (self.conv2 (x)), 2)
        x = x.view (-1, self.num_flat_features (x))
        x = F.selu (self.fc1 (x))
        x = F.selu (self.fc2 (x))
        x = self.fc3 (x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# CUDA for PyTorch
use_cuda = torch.cuda.is_available ()
device = torch.device ("cuda:0" if use_cuda else "cpu")

data = ImageFolder (root = 'gtsrb', transform = transforms.Compose ([
    transforms.Resize ([30,30]),
    transforms.ToTensor (),
    transforms.Normalize (mean = (0, 0, 0), std = (1, 1, 1))
]))

cut_point = int(TEST_SIZE * len (data))
training_data, test_data = torch.utils.data.random_split (
    data,
    [len (data) - cut_point, cut_point]
)

training_data_provider = DataLoader (
    dataset = training_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
    pin_memory = True
)

test_provider = DataLoader (
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle = False,
    pin_memory = True
)

net = Net ()
if use_cuda:
    net.cuda ()
criterion = nn.CrossEntropyLoss ()
optimizer = optim.Adam (net.parameters ())

BATCHES_PER_PRINT = 100

for epoch in range (EPOCHS):
    e_data = iter (training_data_provider)
    print ('E', epoch)
    running_loss = 0.0
    correct = 0.0
    for i, data in enumerate (e_data):
        inputs, labels = data
        inputs, labels = inputs.to (device), labels.to (device)
        optimizer.zero_grad ()
        outputs = net (inputs)
        loss = criterion (outputs, labels)
        loss.backward ()
        optimizer.step ()
        _, max_indices = torch.max (outputs, 1)
        correct += (sum (max_indices == labels) / len(labels)).item ()
        # Print statistics
        running_loss += loss.item ()
        if i % BATCHES_PER_PRINT == BATCHES_PER_PRINT - 1: 
            print (
                '[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / BATCHES_PER_PRINT)
            )
            running_loss = 0.0
            print (
                'accuracy: %.3f' % (correct / BATCHES_PER_PRINT)
            )
            correct = 0.0

print ('Finished Training')

with torch.set_grad_enabled (False):
    running_loss = 0.0
    correct = 0.0
    for local_batch, local_labels in test_provider:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to (device), local_labels.to (device)
        outputs = net (local_batch)
        loss = criterion (outputs, local_labels)
        _, max_indices = torch.max (outputs, 1)
        running_loss += loss.item ()
        correct += (sum (max_indices == local_labels) / len(local_labels)).item ()
    print ('Loss', running_loss / len (test_provider))
    print ('Accuracy', correct / len (test_provider))

"""
print(net)

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

optimizer = optim.SGD(net.parameters(), lr=0.01)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

for i in range(10):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    optimizer.step()
"""

