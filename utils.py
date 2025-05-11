from lib import *
from config import *
import pandas as pd


# def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("device: ", device)
#
#     # Lists to store loss and accuracy for each epoch
#     history = {
#         "epoch": [],
#         "phase": [],
#         "loss": [],
#         "accuracy": []
#     }
#
#     for epoch in range(num_epochs):
#         print("Epoch {}/{}".format(epoch, num_epochs))
#
#         # move network to device(GPU/CPU)
#         net.to(device)
#         torch.backends.cudnn.benchmark = True
#
#         for phase in ["train", "val"]:
#             if phase == "train":
#                 net.train()
#             else:
#                 net.eval()
#
#             epoch_loss = 0.0
#             epoch_corrects = 0
#
#             if (epoch == 0) and (phase == "train"):
#                 continue
#
#             for inputs, labels in tqdm(dataloader_dict[phase]):
#                 # move inputs, labels to GPU/CPU
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # set gradient of optimizer to be zero
#                 optimizer.zero_grad()
#
#                 with torch.set_grad_enabled(phase == "train"):
#                     outputs = net(inputs)
#
#                     outputs = torch.tensor(outputs, dtype=torch.float32)
#                     labels = torch.tensor(labels, dtype=torch.long)
#
#                     loss = criterion(outputs, labels)
#                     _, preds = torch.max(outputs, 1)
#
#                     if phase == "train":
#                         loss.backward()
#                         optimizer.step()
#
#                     epoch_loss += loss.item() * inputs.size(0)
#                     epoch_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
#             epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
#
#             print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))
#
#             # Save results for this epoch and phase
#             history["epoch"].append(epoch)
#             history["phase"].append(phase)
#             history["loss"].append(epoch_loss)
#             history["accuracy"].append(epoch_accuracy.item())
#
#     # Save the model
#     torch.save(net.state_dict(), save_path)
#
#     # Save history to CSV
#     df = pd.DataFrame(history)
#     df.to_csv(csv_path, index=False)
#     print(f"Training history saved to {csv_path}")


def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    history = {
        "epoch": [],
        "phase": [],
        "loss": [],
        "accuracy": []
    }

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        net.to(device)
        torch.backends.cudnn.benchmark = True

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            # Skip first epoch training if desired
            if (epoch == 0) and (phase == "train"):
                continue

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)

                    # 👉 KHÔNG dùng torch.tensor(outputs)
                    outputs = outputs.float()
                    labels = labels.long()

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

            history["epoch"].append(epoch)
            history["phase"].append(phase)
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_accuracy.item())

    torch.save(net.state_dict(), save_path)

    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)
    print(f"Training history saved to {csv_path}")