import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import base64
import tensorflow as tf

def loadImage(base64string):
    if "base64," in base64string:
        base64string = base64string[base64string.index(",")+1:]
    img_bytes = base64.b64decode(base64string)
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    np_arr = np.array(img)
    Cvimg = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    return Cvimg

def preprocessImage(img,td=224):
    shp = tf.cast(tf.shape(img)[1:-1], tf.float32)
    sd  = min(shp)
    scl = td/sd
    nhp = tf.cast(shp*scl, tf.int32)
    img = tf.image.resize(img,nhp)
    img = tf.image.resize_with_crop_or_pad(img, td,td)
    return img

def postprocessImage(img):
    o = (np.squeeze(img)+1.0)*127.5
    o = np.clip(o,0,255).astype(np.uint8)
    o = cv2.cvtColor(o,cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', o)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def cartoon(img_p):
    # loading image
    si = loadImage(img_p)

    psi = preprocessImage(si,td=512)
    psi.shape

    # model dataflow 
    m = '1.tflite'
    i = tf.lite.Interpreter(model_path=m)
    ind = i.get_input_details()
    i.allocate_tensors()
    i.set_tensor(ind[0]['index'],psi)
    i.invoke()

    r = i.tensor(i.get_output_details()[0]['index'])()

    # post process the model output
    base64str = postprocessImage(r)
    return base64str
