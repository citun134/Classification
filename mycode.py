import numpy as np

def random_horizontal_flip(image, p=0.5):
    if np.random.rand() < p:
        return np.fliplr(image)  # Lật ảnh theo chiều ngang
    return image

def to_tensor(image):
    # Chuyển đổi từ H x W x C sang C x H x W
    image = np.transpose(image, (2, 0, 1))  # Đổi trục
    return image.astype(np.float32) / 255.0  # Chuẩn hóa giá trị về [0, 1]

def normalize(tensor, mean, std):
    # Tensor có dạng C x H x W
    for c in range(tensor.shape[0]):  # Duyệt qua từng kênh
        tensor[c] = (tensor[c] - mean[c]) / std[c]
    return tensor


def transform_image(image, resize, mean, std, p=0.5):
    # Resize (giả sử sử dụng OpenCV hoặc PIL)
    # image = cv2.resize(image, (resize, resize))  # Nếu sử dụng OpenCV

    # Random Horizontal Flip
    image = random_horizontal_flip(image, p)

    # To Tensor
    tensor = to_tensor(image)

    # Normalize
    tensor = normalize(tensor, mean, std)

    return tensor


if __name__ == "__main__":

    # Giả sử ảnh là NumPy array có kích thước H x W x C
    image = np.random.randint(0, 256, (3, 3, 3), dtype=np.uint8)

    print(f"original: {image}")

    p = 0.6
    image = random_horizontal_flip(image, p)
    print(f"horizontal: {image}")
    # Thông số
    resize = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # # Chuyển đổi ảnh
    # transformed_image = transform_image(image, resize, mean, std)
    # print(transformed_image.shape)  # Kết quả: (3, 224, 224)

