import matplotlib.pyplot as plt

from lib import *
from config import *
import random

def make_datapath_list(root_path):
    train_path = "train"
    val_path = "val"
    test_path = "test"

    train_target_path = osp.join(root_path, train_path, "**/*.png")
    val_target_path = osp.join(root_path, val_path, "**/*.png")
    test_target_path = osp.join(root_path, test_path, "**/*.png")

    train_list = []
    val_list = []
    test_list = []

    for path in glob.glob(train_target_path):
        train_list.append(path)

    for path in glob.glob(val_target_path):
        val_list.append(path)

    for path in glob.glob(test_target_path):
        test_list.append(path)

    return train_list, val_list, test_list


def make_datapath_list_pro(root_path):
    train_path_normal = osp.join(root_path, r"normal\images_split\train\*.png")
    train_path_abnormal = osp.join(root_path, r"abnormal\images\train\*.png")

    # val_path_normal = osp.join(root_path, "normal/images_split/val/*.png")
    # val_path_abnormal = osp.join(root_path, "abnormal/images/val/*.png")

    train_list = []
    val_list = []

    for path in glob.glob(train_path_normal):
        train_list.append(path)

    for path in glob.glob(train_path_abnormal):
        train_list.append(path)

    # for path in glob.glob(val_path_normal):
    #     val_list.append(path)
    #
    # for path in glob.glob(val_path_abnormal):
    #     val_list.append(path)

    return train_list
    # return train_list, val_list

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train" : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)

class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")  # Đảm bảo luôn là ảnh RGB

        img_transformed = self.transform(img, self.phase)

        # Lấy tên thư mục cha làm label (tương thích mọi hệ điều hành)
        label_name = os.path.basename(os.path.dirname(img_path))

        if label_name == "abnormal":
            label = 0
        elif label_name == "normal":
            label = 1
        else:
            raise ValueError(f"Không xác định được label từ đường dẫn: {img_path}")

        return img_transformed, label

    # def __getitem__(self, idx):
    #     img_path = self.file_list[idx]
    #     img = Image.open(img_path)
    #
    #     img_transformed = self.transform(img, self.phase)
    #
    #     if self.phase == "train":
    #         label = img_path.split("\\")[-2]
    #         # print(f"label train:{label}")
    #     elif self.phase == "val":
    #         label = img_path.split("\\")[-2]
    #     elif self.phase == "test":
    #         label = img_path.split("\\")[-2]
    #         # print(f"label train:{label}")
    #
    #     if label == "abnormal":
    #         label = 0
    #     elif label == "normal":
    #         label = 1
    #
    #     return img_transformed, label

# class MyDataset(data.Dataset):
#     def __init__(self, file_list, transform=None, phase="train"):
#         self.file_list = file_list
#         self.transform = transform
#         self.phase = phase
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         img_path = self.file_list[idx]
#         img = Image.open(img_path)
#         img_transformed = self.transform(img, self.phase)
#
#         # Trích xuất tên thư mục cấp trên từ đường dẫn (lấy "abnormal" hoặc "normal")
#         directory_name = img_path.split(os.sep)[-4]  # Lấy tên thư mục cấp trên của "images"
#         # print(directory_name)
#         # Xác định nhãn dựa trên tên thư mục
#         if directory_name == "abnormal":
#             label = 0  # Bất thường
#         elif directory_name == "normal":
#             label = 1  # Bình thường
#         else:
#             raise ValueError("Thư mục không hợp lệ, không phải 'abnormal' hoặc 'normal'")
#
#         return img_transformed, label

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


if __name__ == "__main__":
    root_path = r"E:\data_labels\classification"
    train_list, val_list, test_list = make_datapath_list(root_path)
    print("Train List:", len(train_list))
    print(train_list[0])
    # print(test_list)

    transform = ImageTransform(resize=resize, mean=mean, std=std)

    test_dataset = MyDataset(test_list, transform, test)
    train_dataset = MyDataset(train_list, transform, train)
    val_dataset = MyDataset(val_list, transform, val)

    index = 100
    unorm = UnNormalize(mean=mean, std=mean)

    # read image
    img_path = test_list[index]
    img_original = Image.open(img_path)

    # image to tensor, label
    img_transform, label = test_dataset.__getitem__(index)

    # # image transform
    img_transformed = transform(img_original, phase=train)
    img_transformed = img_transformed.numpy().transpose(1, 2, 0)
    img_transformed = np.clip(img_transformed, 0, 1)

    print(f"label: {label}")
    print(img_transform.shape)

    plt.subplot(1,3,1)
    plt.imshow(img_original)

    plt.subplot(1, 3, 2)
    plt.imshow(img_transformed)

    plt.subplot(1, 3, 3)
    plt.imshow(unorm(img_transform).permute(1, 2, 0))
    plt.show()

