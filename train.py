import torch

from lib import *
from config import *
from model import *
from data import *
import torch.utils.data as data
from torchvision import models
from datetime import datetime
from utils import *
from mycode import *
from torchvision.models import ResNet50_Weights


n_workers = os.cpu_count()
print("num_workers =", n_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
root_path = r"E:\data_labels\classification"
# train_list, val_list = make_datapath_list(root_path)
train_list, val_list, test_list = make_datapath_list(root_path)

transform = ImageTransform(resize=resize, mean=mean, std=std)

train_dataset = MyDataset(train_list, transform, train)
val_dataset = MyDataset(val_list, transform, val)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {"train": trainloader, "val": valloader}

# test dataloader
batch_iterator = iter(trainloader)
inputs, labels = next(batch_iterator)
print(inputs.size()) # inputs.shape
print(labels)

# model
# model = VGG(conv_arch, 2, 3).to(device)


# use_pretrained = True
# model = models.mobilenet_v3_large(use_pretrained=use_pretrained)


# use_pretrained = ResNet50_Weights.IMAGENET1K_V2
# model = models.resnet50(use_pretrained=use_pretrained)
# model.classifier[3] = nn.Linear(in_features=1280, out_features=2)

use_pretrained = True
net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
net.fc = nn.Linear(in_features=2048, out_features=2)




# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

train_model(model, dataloader_dict, criterion, optimizer, num_epochs)

# training
######################
# def train_one_epoch(epoch_index):
#     running_loss = 0.0
#     last_loss = 0.0
#     running_corrects = 0
#     total_samples = 0
#
#     for i, data in enumerate(tqdm(trainloader, desc=f"Training Epoch {epoch_index+1}", leave=False)):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # Tích lũy loss và tính accuracy cho mỗi batch
#         running_loss += loss.item() * labels.size(0)
#
#         _, preds = torch.max(outputs, 1)
#         running_corrects += torch.sum(preds == labels).item()
#         total_samples += labels.size(0)
#
#     last_loss = running_loss / total_samples
#     epoch_accuracy = running_corrects / total_samples
#     return last_loss, epoch_accuracy
#
#
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# epoch_number = 0
#
# EPOCHS = 5
#
# best_vloss = 1_000_000.
#
# # Thêm vòng lặp cho EPOCHS và tính toán độ chính xác trên tập kiểm tra
# for epoch in range(EPOCHS):
#     print('EPOCH {}:'.format(epoch_number + 1))
#
#     # Huấn luyện
#     model.train(True)
#     avg_loss, train_accuracy = train_one_epoch(epoch_number)
#
#     # Đánh giá trên tập kiểm tra (validation)
#     running_vloss = 0.0
#     running_vcorrects = 0
#     model.eval()
#
#     with torch.no_grad():
#         total_samples = 0
#         for i, vdata in enumerate(tqdm(valloader, desc=f"Validating Epoch {epoch_number+1}", leave=False)):
#             vinputs, vlabels = vdata
#             vinputs, vlabels = vinputs.to(device), vlabels.to(device)
#
#             voutputs = model(vinputs)
#             vloss = criterion(voutputs, vlabels)
#             running_vloss += vloss.item() * vlabels.size(0)
#
#             # Tính accuracy cho tập validation
#             _, vpreds = torch.max(voutputs, 1)
#             running_vcorrects += torch.sum(vpreds == vlabels).item()
#             total_samples += vlabels.size(0)
#
#     avg_vloss = running_vloss / total_samples
#     validation_accuracy = running_vcorrects / total_samples
#     print('LOSS: train {}, valid {}; Accuracy: train {}, valid {}'.format(avg_loss, avg_vloss, train_accuracy,
#                                                                       validation_accuracy))
#
#     # Lưu lại mô hình tốt nhất
#     if avg_vloss < best_vloss:
#         best_vloss = avg_vloss
#         model_path = r"D:\PyCharm\pythonProject\Classification\model\model_{}_{}".format(timestamp, epoch_number)
#         torch.save(model.state_dict(), model_path)
#
#     epoch_number += 1


