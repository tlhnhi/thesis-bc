from flask import Flask, request, redirect, render_template
from PIL import Image
from io import BytesIO
from base64 import b64encode
from numpy import asarray

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

weight_path = "../R50_FPN_3x.pth"
model_name = "faster_rcnn_R_50_FPN_3x"
padding = 0.2
threshold = 0.65

# Init detector before starting server
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = weight_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

padding += 1
detector = DefaultPredictor(cfg)


def img_to_b64(img):
    img_output = BytesIO()
    img.save(img_output, format='JPEG')
    img_bytes = img_output.getvalue()
    return b64encode(img_bytes).decode("utf-8")


def detect(image, detection_classes):
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
        # Skip if detected class is not in detection_classes
        if (classes[pred_classes[i]] not in detection_classes):
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

        # Save predict result
        preds[classes[pred_classes[i]]].append({
            "img": img_to_b64(image.crop((x1, y1, x2, y2))),
            "dscore": round(scores[i], 2)
        })
    return preds


def get_output(img_bytes, detection_classes, classification_groups):
    image = Image.open(BytesIO(img_bytes))
    img_b64 = img_to_b64(image)

    preds = detect(image, detection_classes)

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
        return render_template('result.html', preds=preds, img_b64=img_b64)
    else:
	    return render_template('index.html')