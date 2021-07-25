from flask import Flask, request, redirect, render_template
from PIL import Image
from io import BytesIO
from base64 import b64encode


def get_prediction(img_bytes):
    image = Image.open(BytesIO(img_bytes))

    img_output = BytesIO()
    image.save(img_output, format='JPEG')
    img_bytes = img_output.getvalue()

    image = b64encode(img_bytes).decode("utf-8")

    preds = None

    return preds, image

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
        

        preds, out_img = get_prediction(img_bytes)
        return render_template('result.html', preds=preds, image=out_img)
    else:
	    return render_template('index.html')