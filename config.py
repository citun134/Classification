from lib import *

batch_size = 8
num_epochs = 5

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train = "train"
val = "val"
test = "test"

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

save_path = r"D:\PyCharm\pythonProject\Classification\model\model_class_ep_1.pth"
csv_path= r"D:\PyCharm\pythonProject\Classification\model\training_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
