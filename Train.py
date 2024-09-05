import torch
import torch.nn as nn
from vit_pytorch import SimpleViT
from torch.utils.data import TensorDataset, DataLoader
# torch.set_default_device('cuda')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleViT(
    image_size=2048,
    patch_size=128,
    num_classes=351,# 256/32
    dim=1024,
    depth = 8,
    heads = 4,
    mlp_dim=2048,
)
# model = model.to(torch.float16)
print(model.named_modules())

train_data = torch.load("train_data.pt")
targets_data = torch.load("targets_data2.pt")
print(train_data.dtype)
targets_data = torch.argmax(targets_data, dim=-1)
print(targets_data[:10])
# torch.from_numpy()
train_data = train_data.transpose(1, -1)
# train_data = train_data.to(torch.float)
dataset = TensorDataset(train_data[:10], targets_data[:10])

train_dataloader = DataLoader(dataset, batch_size=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lossfunc = nn.CrossEntropyLoss()

model = model.to(device)

epochs = 1000

c=5
# start = time.time()
val=0
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')



    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):


        x = batch[0].to(device)
        y = batch[1].to(device)

        # x = x.to(torch.float16)
        # y = y.to(torch.int16)

        print(x.dtype, y.dtype)
        # print(b_labels)

        optimizer.zero_grad()

        outputs = model(x)

        loss = lossfunc(outputs, y)

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.

        loss.backward()

        optimizer.step()
        torch.cuda.empty_cache()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    if avg_train_loss < c:
      c = avg_train_loss
      torch.save(model.state_dict(),"ViTModel.pt")
      print(f"saved {c}")
    print(f"Average train loss:{avg_train_loss}")
    if avg_train_loss < 0.05:
      print(f"Yay Got there at Epoch:{epoch_i} Step:{step} Average Loss:{avg_train_loss} Loss{loss.item()}")
      break