import numpy as np
import torch.nn as nn

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




class Sequential:
    def __init__(self, *layers):
        """
        Sequential layer similar to PyTorch Sequential.
        Layers are passed in order of execution.
        """
        self.layers = layers  # Danh sách các lớp được truyền vào

    def forward(self, x):
        """
        Forward pass qua tất cả các lớp theo thứ tự.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)


# class Sequential:
#     def __init__(self, *layers):
#         """
#         Sequential layer similar to PyTorch Sequential.
#         Layers are passed in order of execution.
#         """
#         self.layers = layers  # Danh sách các lớp được truyền vào
#
#     def forward(self, x):
#         """
#         Forward pass qua tất cả các lớp theo thứ tự.
#         """
#         for layer in self.layers:
#             x = layer.forward(x)
#         return x
#
#     def __call__(self, x):
#         return self.forward(x)
#
#     def parameters(self):
#         """
#         Lấy tất cả các tham số (weights, biases, ...) và tên của các layer.
#         """
#         params = []
#         for layer in self.layers:
#             layer_name = type(layer).__name__
#             if hasattr(layer, "parameters") and callable(layer.parameters):
#                 layer_params = layer.parameters()
#                 params.append((layer_name, layer_params))
#         return params


class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        return np.dot(x, self.W) + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def forward(self, x):
        """
        Forward pass cho ReLU activation.
        """
        self.input = x
        return np.maximum(0, x)


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        """
        Max Pooling layer.
        """
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        Forward pass cho MaxPool2D layer.
        """
        batch_size, height, width, channels = x.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, out_height, out_width, channels))

        for i in range(out_height):
            for j in range(out_width):
                x_patch = x[:, i * self.stride:i * self.stride + self.kernel_size,
                          j * self.stride:j * self.stride + self.kernel_size, :]
                out[:, i, j, :] = np.max(x_patch, axis=(1, 2))
        return out


class Flatten:
    def forward(self, x):
        """
        Flatten layer: Chuyển tensor nhiều chiều thành vector 1 chiều.
        """
        return x.reshape(x.shape[0], -1)

class Dropout:
    def __init__(self, p=0.5):
        """
        Khởi tạo lớp Dropout.

        Parameters:
            p (float): Xác suất tắt (drop) một phần tử. p nằm trong khoảng [0, 1].
        """
        assert 0 <= p <= 1, "Xác suất p phải nằm trong khoảng [0, 1]."
        self.p = p
        self.training = True  # Mặc định đang trong chế độ training

    def forward(self, x):
        """
        Thực hiện Dropout trên input x.

        Parameters:
            x (numpy.ndarray): Input dữ liệu (batch_size, ...).

        Returns:
            numpy.ndarray: Output sau khi thực hiện Dropout.
        """
        if not self.training or self.p == 0:  # Không làm gì nếu đang trong chế độ eval hoặc p=0
            return x

        # Tạo mask với xác suất giữ lại (1 - p)
        self.mask = np.random.rand(*x.shape) > self.p

        # Chia tỷ lệ giá trị còn lại để đảm bảo giá trị trung bình không thay đổi
        return x * self.mask / (1 - self.p)

    def __call__(self, x):
        """
        Cho phép gọi lớp như một hàm.
        """
        return self.forward(x)

    def train(self):
        """
        Chuyển sang chế độ training.
        """
        self.training = True

    def eval(self):
        """
        Chuyển sang chế độ evaluation (tắt Dropout).
        """
        self.training = False


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        """
        Khởi tạo lớp Conv2D.

        Parameters:
            in_channels (int): Số kênh đầu vào.
            out_channels (int): Số kênh đầu ra (số bộ lọc).
            kernel_size (int): Kích thước của bộ lọc (giả định vuông).
            stride (int): Bước nhảy.
            padding (int): Padding xung quanh input.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Khởi tạo trọng số và bias ngẫu nhiên
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.random.randn(out_channels)

    def forward(self, x):
        """
        Thực hiện phép tích chập trên dữ liệu đầu vào x.

        Parameters:
            x (numpy.ndarray): Input có kích thước (batch_size, in_channels, H, W).

        Returns:
            numpy.ndarray: Output sau khi thực hiện tích chập.
        """
        # Lấy kích thước input
        batch_size, in_channels, H, W = x.shape
        assert in_channels == self.in_channels, "Số kênh đầu vào không khớp!"

        # Thêm padding vào input
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Tính kích thước output
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Khởi tạo output
        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        # Thực hiện phép tích chập
        for b in range(batch_size):  # Duyệt qua batch
            for oc in range(self.out_channels):  # Duyệt qua các bộ lọc
                for h in range(H_out):  # Duyệt qua chiều cao
                    for w in range(W_out):  # Duyệt qua chiều rộng
                        # Lấy vùng nhỏ tương ứng với kernel
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        x_slice = x_padded[b, :, h_start:h_end, w_start:w_end]

                        # Tính tích vô hướng giữa bộ lọc và vùng input
                        output[b, oc, h, w] = np.sum(x_slice * self.weight[oc]) + self.bias[oc]

        return output

    def __call__(self, x):
        """
        Cho phép gọi lớp như một hàm.
        """
        return self.forward(x)




if __name__ == "__main__":

    # Giả sử ảnh là NumPy array có kích thước H x W x C
    image = np.random.randint(0, 256, (3, 3, 3), dtype=np.uint8)

    # print(f"original: {image}")

    p = 0.6
    image = random_horizontal_flip(image, p)
    # print(f"horizontal: {image}")
    # Thông số
    resize = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # # Chuyển đổi ảnh
    # transformed_image = transform_image(image, resize, mean, std)
    # print(transformed_image.shape)  # Kết quả: (3, 224, 224)



    # Số lớp và đầu ra
    np.random.seed(42)  # Đặt seed để tái lập kết quả

    # Tạo input mẫu: Batch size 1, 4x4 ảnh với 1 channel
    input_image = np.random.randn(2, 4, 4, 1)

    # Xây dựng Sequential model
    model = Sequential(
        MaxPool2D(kernel_size=2, stride=2),  # Max Pooling
        Flatten(),  # Flatten
        Linear(4, 2),  # Fully Connected Layer (4 input, 2 output)
        ReLU()  # Activation Function
    )

    # Forward pass
    output = model(input_image)
    print("Output shape:", output.shape)
    print("Output values:\n", output)

    # model = Sequential(
    #     Linear(10, 20),
    #     Linear(20, 30)
    # )
    #
    # params = model.parameters()
    # for layer_name, layer_params in params:
    #     print(f"Layer: {layer_name}")
    #     for param in layer_params:
    #         print(param.shape)

