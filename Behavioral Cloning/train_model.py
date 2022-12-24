import torch
from torch import nn

import torchvision
from torchvision import models as models

from config import * 
from prepare_data import Data_Generator


datagen = Data_Generator()
model = models.alexnet(pretrained=True)

for param in list(model.parameters())[:-6]:
    param.requires_grad = False

model.classifier[4] = nn.Linear(in_features=4096, out_features=2048)
model.classifier[6] = nn.Linear(in_features=2048, out_features=1)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), LR)

images, angles = next(datagen.data_generator())
for epoch in range(1):
    for step in range(datagen.steps_per_epoch):
        images, angles = next(datagen.data_generator())
        images = torch.Tensor(images, device=DEVICE).view(-1, 3, *datagen.image_shape[:-1])
        angles = torch.Tensor(angles, device=DEVICE)

        optimizer.zero_grad()        
        labels = model.forward(images)

        l = loss(labels, angles)
        l.backward()
        optimizer.step()
    LOG_MESSAGE = F"Epoch: {epoch+1} / {EPOCHS: }\n".ljust(20)
    LOG_MESSAGE += F"[{'='.ljust(step, '=')}>{' '.ljust((datagen.steps_per_epoch-step))}]"
    LOG_MESSAGE += F"\t Loss: {l.item()}"
    print(LOG_MESSAGE)

torch.save(model.state_dict(), MODEL_PATH)