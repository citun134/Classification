import torch
from tqdm import tqdm
import pandas as pd
from lib import *
from data import *
from torchvision import models
import torch.nn as nn


def evaluate_model(net, dataloader, criterion, csv_path="evaluation_results.csv"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()  # Đặt mô hình vào chế độ đánh giá

    test_loss = 0.0
    test_corrects = 0
    total_samples = 0

    # Tắt gradient khi đánh giá để tiết kiệm bộ nhớ
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            # Di chuyển dữ liệu vào thiết bị (GPU/CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Dự đoán
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Tính toán loss và số lượng dự đoán đúng
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    # Tính toán độ chính xác và loss trung bình trên tập test
    avg_loss = test_loss / total_samples
    accuracy = test_corrects.double() / total_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Lưu kết quả vào file CSV
    results = {
        "Test Loss": [avg_loss],
        "Test Accuracy": [accuracy.item()]
    }
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return avg_loss, accuracy


if __name__ == "__main__":
    root_path = r"E:\data_labels\classification"
    # Tạo danh sách dữ liệu
    train_list, val_list, test_list = make_datapath_list(root_path)

    # Chuyển đổi dữ liệu
    transform = ImageTransform(resize=resize, mean=mean, std=std)
    test_list = MyDataset(test_list, transform, test)
    testloader = torch.utils.data.DataLoader(test_list, batch_size=batch_size, shuffle=False)

    # Khởi tạo mô hình và tải trọng số
    use_pretrained = True
    net = models.mobilenet_v3_large(pretrained=use_pretrained)
    net.classifier[3] = nn.Linear(in_features=1280, out_features=2)

    # Load trọng số từ file
    model_path = r"D:\PyCharm\pythonProject\Classification\model\model_class_ep_1.pth"
    net.load_state_dict(
        torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

    # Định nghĩa hàm loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    csv_path = r"D:\PyCharm\pythonProject\Classification\model\test_8_11_2024.csv"

    # Gọi hàm evaluate_model để đánh giá mô hình
    test_loss, test_accuracy = evaluate_model(net, testloader, criterion, csv_path)
