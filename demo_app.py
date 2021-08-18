from flask import Flask, request, redirect, render_template
from PIL import Image
from io import BytesIO
from base64 import b64encode
from numpy import asarray

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import torch
from torchvision import models, transforms

detector_weight = "./weights/R50_FPN_3x.pth"
detector_model = "faster_rcnn_R_50_FPN_3x"
padding = 0.2
threshold = 0.65

rbc_weight = "./weights/densenet_rbc.pt"
wbc_weight = "./weights/densenet_wbc.pt"


# Init detector before starting server
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{detector_model}.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = detector_weight
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

if not torch.cuda.is_available(): # Use cpu if cuda isnt available
    print("\nUsing cpu for detection")
    cfg.MODEL.DEVICE="cpu"

padding += 1
print("\nLoading detector...")
detector = DefaultPredictor(cfg)


# Init classifiers before starting server
if not torch.cuda.is_available() or torch.cuda.memory_reserved(0)/1024**3 < 2: # Use cpu if cuda isnt available
    print("\nUsing cpu for classification\n")
    cls_device = "cpu"
else:
    cls_device = "cuda"
cls_device = torch.device(cls_device)

def init_classifier(weight, cls_num):
    model = models.densenet121()
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, cls_num)
    model.load_state_dict(torch.load(weight, map_location=cls_device))
    model.eval()
    model = model.to(cls_device)
    return model

classifier = {}
print("Loading rbc classifier...")
classifier["rbc"] = init_classifier(rbc_weight, 3)
print("Loading wbc classifier...\n")
classifier["wbc"] = init_classifier(wbc_weight, 5)

def img_to_b64(img):
    img_output = BytesIO()
    img.save(img_output, format='JPEG')
    img_bytes = img_output.getvalue()
    return b64encode(img_bytes).decode("utf-8")

def classify(img, group="wbc"):
    loader = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    cls_classes = {
        "wbc": ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil'],
        "rbc": ["tròn", "dài", "khác"]
    }

    image_tensor = loader(img).float()
    inp = image_tensor.unsqueeze(0)
    inp = inp.to(cls_device)
    
    if group in classifier:
        out = classifier[group](inp)
        confs = torch.nn.functional.softmax(out[0], dim=0)
        _, pred = torch.max(out, 1)
        return cls_classes[group][pred[0]], confs[pred[0]].item()


def detect(image, detection_classes, classification_groups):
    img_arr = asarray(image) # From PIL to numpy array
    img_arr = img_arr[:, :, ::-1] # RGB to BGR

    outputs = detector(img_arr)
    ih, iw, _ = img_arr.shape

    pred_classes = outputs['instances'].pred_classes.cpu().tolist()
    scores = outputs['instances'].scores.cpu().tolist()
    classes = ["", "platelet", "rbc", "wbc"] # Classes from weight

    # Init a dict to store predict result
    preds = {}
    for cls in detection_classes:
        preds[cls] = []
    for i, box in enumerate(outputs['instances'].pred_boxes.__iter__()):
        pred_cls = classes[pred_classes[i]]

        # Skip if detected class is not in detection_classes
        if pred_cls not in detection_classes:
            continue

        # Get original shape of detected cell
        x1, y1, x2, y2 = box.cpu().numpy()
        h, w = y2-y1, x2-x1

        # Calculate crop shape
        dh, dw = h*padding, w*padding

        # Calculate x1, y1, x2, y2 to crop
        dh, dw = (dh-h)/2, (dw-w)/2
        x1, x2 = x1 - dh, x2 + dh
        y1, y2 = y1 - dw, y2 + dw

        # Add bias to prevent box outside image
        if (x1 < 0):
            x2 -= x1
            x1 = 0
        if (x2 > iw):
            x1 -= (x2 - iw)
            x2 = iw
        if (y1 < 0):
            y2 -= y1
            y1 = 0
        if (y2 > ih):
            y1 -= (y2 - ih)
            y2 = ih

        # Type casting
        x1, y1, x2, y2 = int(x1), int(y1), int(x2 + 0.9), int(y2 + 0.9)

        # Crop image
        cropped = image.crop((x1, y1, x2, y2))

        # Save predict result
        preds[pred_cls].append({
            "img": img_to_b64(cropped),
            "dscore": round(scores[i], 2)
        })

        if pred_cls in classification_groups:
            cls, conf = classify(cropped, pred_cls)
            preds[pred_cls][-1]["cls"] = cls
            preds[pred_cls][-1]["cscore"] = round(conf, 2)
    return preds


def get_output(img_bytes, detection_classes, classification_groups):
    image = Image.open(BytesIO(img_bytes))
    img_b64 = img_to_b64(image)

    preds = detect(image, detection_classes, classification_groups)

    return preds, img_b64


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        post_file = request.files.get('file')
        if not post_file:
            return redirect(request.url)
        img_bytes = post_file.read()
        detection_classes = request.form.getlist('detection')
        classification_groups = request.form.getlist('classification')
        print("detection_classes", detection_classes)
        print("classification_groups", classification_groups)

        preds, img_b64 = get_output(img_bytes, detection_classes, classification_groups)
        return render_template('result.html', preds=preds, img_b64=img_b64, cls_groups=classification_groups)
    else:
	    return render_template('index.html')