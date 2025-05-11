import numpy as np
import tqdm
class Linear:
    def __init__(self, input_dim, output_dim):
        """
        Lớp Fully Connected (Dense Layer).
        """
        self.W = np.random.randn(input_dim, output_dim) * 0.01  # Khởi tạo trọng số
        self.b = np.zeros((1, output_dim))  # Khởi tạo bias

    def forward(self, x):
        """
        Forward pass cho Linear layer.
        """
        self.input = x  # Lưu input để dùng cho backward (nếu cần)
        return np.dot(x, self.W) + self.b

    def __call__(self, x):
        return self.forward(x)

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        """
        Khởi tạo lớp Conv2D.
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
        Thực hiện phép tích chập.
        """
        # Lấy kích thước input
        batch_size, in_channels, H, W = x.shape
        assert in_channels == self.in_channels, f"Số kênh đầu vào không khớp! Expect {self.in_channels}, got {in_channels}"

        # Thêm padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Kích thước output
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Khởi tạo output
        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        # Thực hiện tích chập
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        x_slice = x_padded[b, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                        output[b, oc, h, w] = np.sum(x_slice * self.weight[oc]) + self.bias[oc]
        return output

    def __call__(self, x):
        return self.forward(x)


class MaxPool2D:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, H, W = x.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, channels, H_out, W_out))

        for b in range(batch_size):
            for c in range(channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        x_slice = x[b, c, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                        output[b, c, h, w] = np.max(x_slice)
        return output

    def __call__(self, x):
        return self.forward(x)


class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def __call__(self, x):
        return self.forward(x)


class Flatten:
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

    def __call__(self, x):
        return self.forward(x)


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)


# class VGG:
#     def __init__(self, conv_arch, num_classes=10, in_channels=1):
#         self.conv_blks = self._make_conv_layers(conv_arch, in_channels)
#         self.fc = Sequential(
#             Flatten(),
#             Linear(512 * 7 * 7, 4096),
#             ReLU(),
#             Linear(4096, 4096),
#             ReLU(),
#             Linear(4096, num_classes)
#         )
#
#     def _vgg_block(self, num_convs, in_channels, out_channels):
#         layers = []
#         for _ in range(num_convs):
#             layers.append(Conv2D(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(ReLU())
#             in_channels = out_channels
#         layers.append(MaxPool2D(kernel_size=2, stride=2))
#         return Sequential(*layers)
#
#     def _make_conv_layers(self, conv_arch, in_channels):
#         layers = []
#         for num_convs, out_channels in conv_arch:
#             layers.append(self._vgg_block(num_convs, in_channels, out_channels))
#             in_channels = out_channels
#         return Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv_blks(x)
#         x = self.fc(x)
#         return x

class VGG:
    def __init__(self, conv_arch, num_classes=10, in_channels=1):
        self.conv_blks = self._make_conv_layers(conv_arch, in_channels)
        self.fc = Sequential(
            Flatten(),
            Linear(512 * 7 * 7, 4096),
            ReLU(),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, num_classes)
        )

    def _vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(Conv2D(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(ReLU())
            in_channels = out_channels
        layers.append(MaxPool2D(kernel_size=2, stride=2))
        return Sequential(*layers)

    def _make_conv_layers(self, conv_arch, in_channels):
        layers = []
        for num_convs, out_channels in conv_arch:
            layers.append(self._vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        return Sequential(*layers)

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.fc(x)
        return x

    def parameters(self):
        """
        Trả về danh sách tất cả các tham số (weights và biases) trong model.
        """
        params = []
        for layer in self.conv_blks.layers + self.fc.layers:
            if isinstance(layer, (Linear, Conv2D)):
                params.append(layer.W)
                params.append(layer.b)
        return params

    def __call__(self, x):
        return self.forward(x)

# CrossEntropyLoss
class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-9  # Để tránh log(0)

    def forward(self, y_pred, y_true):
        """
        Tính toán CrossEntropyLoss.
        :param y_pred: (batch_size, num_classes) - Xác suất dự đoán (sau softmax)
        :param y_true: (batch_size, ) - Nhãn thực (dạng index, không phải one-hot)
        :return: scalar - loss trung bình
        """
        batch_size = y_pred.shape[0]
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)  # Clip giá trị để tránh log(0)

        # One-hot encoding y_true
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(batch_size), y_true] = 1

        # Tính loss
        loss = -np.sum(y_true_one_hot * np.log(y_pred)) / batch_size
        return loss

    def backward(self, y_pred, y_true):
        """
        Đạo hàm loss với y_pred để truyền ngược.
        :param y_pred: (batch_size, num_classes)
        :param y_true: (batch_size, )
        :return: gradient cho y_pred
        """
        batch_size = y_pred.shape[0]
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        grad = y_pred.copy()
        grad[np.arange(batch_size), y_true] -= 1
        grad = grad / batch_size
        return grad


# Adam Optimizer
class Adam:
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam optimizer.
        :param model: Model chứa các tham số (weights và biases)
        :param lr: Learning rate
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Step counter

        # Tự động lấy tham số từ model
        self.params = model.parameters()
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        """
        Cập nhật các tham số dựa trên gradients.
        :param grads: Gradients tương ứng với params
        """
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Cập nhật moment và velocity
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Cập nhật tham số
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Hàm train
def train(model, data_loader, loss_fn, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            # Forward pass
            logits = model(x)  # Kết quả dự đoán (trước softmax)
            print(logits)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Chống overflow
            probs /= np.sum(probs, axis=1, keepdims=True)

            # Tính loss
            loss = loss_fn.forward(probs, y)
            total_loss += loss
            print(total_loss)

            # Backward pass
            grad = loss_fn.backward(probs, y)
            grads = []

            # Tính gradient cho từng layer
            for layer in reversed(model.fc.layers + model.conv_blks.layers):
                if isinstance(layer, Linear):
                    dW = np.dot(layer.input.T, grad)
                    db = np.sum(grad, axis=0, keepdims=True)
                    grads.extend([dW, db])
                    grad = np.dot(grad, layer.W.T)
                elif isinstance(layer, ReLU):
                    grad = grad * (layer.input > 0)
                elif isinstance(layer, Conv2D):
                    # Simple backward pass for Conv2D
                    grad_w = np.zeros_like(layer.weight)
                    grad_b = np.zeros_like(layer.bias)
                    for i in range(layer.weight.shape[0]):
                        grad_w[i] = np.sum(grad[:, i, :, :, None, None] * layer.input, axis=(0, 2, 3))
                        grad_b[i] = np.sum(grad[:, i])
                    grads.extend([grad_w, grad_b])

            # Cập nhật trọng số
            optimizer.step(grads)

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # conv_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    # model = VGG(conv_arch, num_classes=10, in_channels=1)
    #
    # input_data = np.random.randn(1, 1, 224, 224)
    # output = model.forward(input_data)
    # print("Dự đoán:", output)

    # Hyperparameters
    batch_size = 2
    num_epochs = 5
    num_classes = 10
    input_channels = 1
    img_size = 28

    # Model, loss và optimizer
    conv_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    model = VGG(conv_arch, num_classes=num_classes, in_channels=input_channels)
    print(model)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model, lr=0.001)

    x_train = np.random.randn(4, 3, 224, 224)  # Một batch của 32 hình ảnh RGB
    y_train = np.random.randint(0, 10, size=(32,))  # Nhãn ngẫu nhiên cho các hình ảnh
    print(x_train)

    # Train
    train(model, [(x_train, y_train)], loss_fn, optimizer, num_epochs=num_epochs)