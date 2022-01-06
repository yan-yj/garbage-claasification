import torchvision.transforms as transform
from torchvision import models
from PIL import Image

# Flask
from flask import Flask, request, render_template, jsonify

import torch
import torch.nn as nn

# 工具
import numpy as np
from util import base64_to_pil


# 创建实例
app = Flask(__name__)

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x


with open('dir_label.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    labels = list(map(lambda x: x.strip().split('\t'), labels))

val_tf = transform.Compose([
    transform.Resize(224),
    transform.ToTensor(),
])

# 图像预处理
def padding_black(img):
    w, h = img.size
    scale = 224. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = 224
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img

# 模型预测
def model_predict(path):
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 214)
    # 加载训练好的模型
    checkpoint = torch.load('models/model', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    img = Image.open(path)
    img = img.resize((224,224))
    img = padding_black(img)
    img = val_tf(img)
    img = np.array(img)
    img = img.reshape(224,224,3)
    img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)

    pred = model(img)
    pred = pred.data.cpu().numpy()[0]
    score = softmax(pred)
    pred_id = np.argmax(score)
    return labels[pred_id][0]


@app.route('/', methods=['GET'])
def index():
    # 默认界面
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 从请求中获取文件
        img = base64_to_pil(request.json)

        # 保存文件
        img.save("./uploads/image.png")

        # 预测
        preds = model_predict("./uploads/image.png")
        return jsonify(result=preds)

    return None


if __name__ == '__main__':
    app.run(port=5002, debug=True)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
