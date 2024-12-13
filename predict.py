from lib import *
from model import *
from config import *
from data import *
import timm
from PIL import Image, ImageDraw, ImageFont

class_index = ["abnormal", "normal"]
# save_path = r"D:\PyCharm\pythonProject\Classification\model\efficientnet\model_efficientnet_9_11.pth"
save_path = r"D:\PyCharm\pythonProject\Classification\model\mobilenet\model_mobilenet_8_11.pth"

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, output):
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.class_index[max_id]
        return predicted_label

def load_model(net, model_path):
    load_weights = torch.load(model_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)

    return net

def predict(img):
    #prepare net
    use_pretrained = True
    net = models.mobilenet_v3_large(use_pretrained=use_pretrained)
    net.classifier[3] = nn.Linear(in_features=1280, out_features=2)

    # NUM_CLASSES = 2
    #
    # class EfficientNet_V2(nn.Module):
    #     def __init__(self, n_out):
    #         super(EfficientNet_V2, self).__init__()
    #         # Define model
    #         self.effnet = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=n_out)
    #
    #     def forward(self, x):
    #         return self.effnet(x)
    #
    # net = EfficientNet_V2(NUM_CLASSES)
    net.eval()
    with torch.no_grad():

        #prepare model
        model = load_model(net, save_path)

        #prepare input img
        transform = ImageTransform(resize, mean, std)
        img = transform(img, phase=test)
        img = img.unsqueeze_(0)

        #predict
        output = model(img)
        predictor = Predictor(class_index)
        response = predictor.predict_max(output)

    return response

if __name__ == "__main__":
    link_normal = r"E:\data_labels\classification\test\normal\data16_18.png"
    link_abnormal = r"E:\data_labels\classification\test\abnormal\data1_39.png"
    # link = r"E:\data\data_root\DSC_0043.JPG" # normal
    # link = r"E:\data\data_root\DSC_0008.JPG" # abnormal

    # Mở ảnh
    img_normal = Image.open(link_normal)
    img_abnormal = Image.open(link_abnormal)


    # Tạo subplot để vẽ 2 ảnh cạnh nhau
    fig, axs = plt.subplots(1, 2)

    # Dự đoán nhãn cho ảnh bình thường
    label_normal = predict(img_normal)
    draw_normal = ImageDraw.Draw(img_normal)
    font_size = 70  # Cỡ chữ
    font = ImageFont.truetype("arial.ttf", font_size)
    text_normal = f"Predicted label: {label_normal}"

    # Vị trí văn bản
    text_position_normal = (10, 10)

    # Vẽ nền cho văn bản
    text_bbox_normal = draw_normal.textbbox(text_position_normal, text_normal, font=font)
    background_position_normal = (
        text_bbox_normal[0] - 5,
        text_bbox_normal[1] - 5,
        text_bbox_normal[2] + 5,
        text_bbox_normal[3] + 5
    )

    # Vẽ hình chữ nhật trắng làm nền
    draw_normal.rectangle(background_position_normal, fill=(255, 255, 255))

    # Vẽ văn bản lên ảnh
    draw_normal.text(text_position_normal, text_normal, fill=(0, 0, 0), font=font)

    # Dự đoán nhãn cho ảnh bất thường
    label_abnormal = predict(img_abnormal)
    draw_abnormal = ImageDraw.Draw(img_abnormal)
    text_abnormal = f"Predicted label: {label_abnormal}"

    # Vị trí văn bản
    text_position_abnormal = (10, 10)

    # Vẽ nền cho văn bản
    text_bbox_abnormal = draw_abnormal.textbbox(text_position_abnormal, text_abnormal, font=font)
    background_position_abnormal = (
        text_bbox_abnormal[0] - 5,
        text_bbox_abnormal[1] - 5,
        text_bbox_abnormal[2] + 5,
        text_bbox_abnormal[3] + 5
    )

    # Vẽ hình chữ nhật trắng làm nền
    draw_abnormal.rectangle(background_position_abnormal, fill=(255, 255, 255))

    # Vẽ văn bản lên ảnh
    draw_abnormal.text(text_position_abnormal, text_abnormal, fill=(0, 0, 0), font=font)

    # Vẽ ảnh bình thường
    axs[0].imshow(img_normal)
    axs[0].axis("off")
    axs[0].set_title("Normal Image")

    # Vẽ ảnh bất thường
    axs[1].imshow(img_abnormal)
    axs[1].axis("off")
    axs[1].set_title("Abnormal Image")

    # Hiển thị các ảnh
    plt.show()