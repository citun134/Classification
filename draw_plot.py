import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv(r'E:\code\Classification\model\test_restnet_13_11_2024.csv')

# Lọc dữ liệu cho các phase 'train' và 'val'
train_df = df[df['phase'] == 'train']
val_df = df[df['phase'] == 'val']

# Vẽ đồ thị Loss
plt.figure(figsize=(12, 5))

# Đồ thị Loss
plt.subplot(1, 2, 1)
plt.plot(train_df['epoch'], train_df['loss'], label='Train Loss', color='blue')
plt.plot(val_df['epoch'], val_df['loss'], label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Đồ thị Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_df['epoch'], train_df['accuracy'], label='Train Accuracy', color='blue')
plt.plot(val_df['epoch'], val_df['accuracy'], label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
