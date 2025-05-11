import torch
from tqdm import tqdm
import pandas as pd
from lib import *
from data import *
from torchvision import models
import torch.nn as nn
import timm
from torchvision.models import inception_v3, Inception_V3_Weights

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, r2_score


# def evaluate_model(net, dataloader, criterion, csv_path="evaluation_results.csv"):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net.to(device)
#     net.eval()  # Đặt mô hình vào chế độ đánh giá
#
#     test_loss = 0.0
#     test_corrects = 0
#     total_samples = 0
#
#     # Tắt gradient khi đánh giá để tiết kiệm bộ nhớ
#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader):
#             # Di chuyển dữ liệu vào thiết bị (GPU/CPU)
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             # Dự đoán
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#
#             # Tính toán loss và số lượng dự đoán đúng
#             test_loss += loss.item() * inputs.size(0)
#             _, preds = torch.max(outputs, 1)
#             test_corrects += torch.sum(preds == labels.data)
#             total_samples += labels.size(0)
#
#     # Tính toán độ chính xác và loss trung bình trên tập test
#     avg_loss = test_loss / total_samples
#     accuracy = test_corrects.double() / total_samples
#
#     print(f"Test Loss: {avg_loss:.4f}")
#     print(f"Test Accuracy: {accuracy:.4f}")
#
#     # Lưu kết quả vào file CSV
#     results = {
#         "Test Loss": [avg_loss],
#         "Test Accuracy": [accuracy.item()]
#     }
#     df = pd.DataFrame(results)
#     df.to_csv(csv_path, index=False)
#     print(f"Results saved to {csv_path}")
#
#     return avg_loss, accuracy

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, r2_score
import numpy as np
import time


def evaluate_model(net, dataloader, criterion, csv_path="evaluation_results.csv"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()  # Đặt mô hình vào chế độ đánh giá

    test_loss = 0.0
    test_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    total_inference_time = 0.0  # Tổng thời gian inference

    # Tắt gradient khi đánh giá để tiết kiệm bộ nhớ
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            # Di chuyển dữ liệu vào thiết bị (GPU/CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Đo thời gian inference
            start_time = time.time()  # Bắt đầu đo thời gian
            outputs = net(inputs)     # Dự đoán
            end_time = time.time()    # Kết thúc đo thời gian

            # Tính inference time cho batch hiện tại
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time

            # Tính toán loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            # Thu thập nhãn thực tế và dự đoán
            test_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính toán các chỉ số cơ bản
    avg_loss = test_loss / total_samples
    accuracy = test_corrects.double() / total_samples

    # Tính các chỉ số bổ sung bằng scikit-learn
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    # Tính inference time trung bình cho mỗi mẫu
    avg_inference_time_per_sample = total_inference_time / total_samples  # giây/mẫu
    total_batches = len(dataloader)
    avg_inference_time_per_batch = total_inference_time / total_batches  # giây/batch

    # In các kết quả
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Inference Time per Sample: {avg_inference_time_per_sample:.6f} seconds")
    print(f"Average Inference Time per Batch: {avg_inference_time_per_batch:.4f} seconds")

    # Lưu kết quả vào file CSV
    results = {
        "Test Loss": [avg_loss],
        "Test Accuracy": [accuracy.item()],
        "Precision": [precision],
        "Recall": [recall],
        "F1-Score": [f1],
        "R2": [r2],
        "Total Inference Time (s)": [total_inference_time],
        "Avg Inference Time per Sample (s)": [avg_inference_time_per_sample],
        "Avg Inference Time per Batch (s)": [avg_inference_time_per_batch]
    }
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return avg_loss, accuracy, precision, recall, f1, r2, conf_matrix, total_inference_time, avg_inference_time_per_sample

# Ví dụ cách gọi hàm
if __name__ == "__main__":
    root_path = r"E:\data_labels\classification"
    # Tạo danh sách dữ liệu
    train_list, val_list, test_list = make_datapath_list(root_path)

    # Chuyển đổi dữ liệu
    transform = ImageTransform(resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Ví dụ
    test_list = MyDataset(test_list, transform, phase='test')  # Đảm bảo 'test' thay vì test
    testloader = torch.utils.data.DataLoader(test_list, batch_size=32, shuffle=False)

    # # Khởi tạo mô hình và tải trọng số
    # use_pretrained = True
    # net = models.mobilenet_v3_large(pretrained=use_pretrained)
    # net.classifier[3] = nn.Linear(in_features=1280, out_features=2)

    # use_pretrained = True
    # net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
    # net.fc = nn.Linear(in_features=2048, out_features=2)

    # NUM_CLASSES = 2
    # class EfficientNet_V2(nn.Module):
    #     def __init__(self, n_out):
    #         super(EfficientNet_V2, self).__init__()
    #         # Define model
    #         self.effnet = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=n_out)
    #
    #     def forward(self, x):
    #         return self.effnet(x)
    #
    #
    # net = EfficientNet_V2(NUM_CLASSES)

    NUM_CLASSES = 2
    class InceptionV3(nn.Module):
        def __init__(self, n_out):
            super(InceptionV3, self).__init__()
            # Define model
            self.inception = timm.create_model('inception_v3', pretrained=True, num_classes=n_out)

        def forward(self, x):
            return self.inception(x)


    # Khởi tạo mô hình
    net =  timm.create_model('inception_v3', pretrained=True, num_classes=NUM_CLASSES)

    # Load trọng số từ file
    model_path = r"E:\code\Classification\model\model_inception_v3_13_12.pth"
    net.load_state_dict(
        torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

    # Định nghĩa hàm loss
    criterion = nn.CrossEntropyLoss()

    csv_path = r"D:\PyCharm\pythonProject\Classification\model\inception_v3.csv"

    # Gọi hàm đánh giá mô hình
    avg_loss, accuracy, precision, recall, f1, r2, conf_matrix, total_time, avg_time = evaluate_model(
        net, testloader, criterion, csv_path
    )