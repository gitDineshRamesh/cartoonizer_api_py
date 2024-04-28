import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import base64
import tensorflow.lite as lite

def loadImage(base64string):
    if "base64," in base64string:
        base64string = base64string[base64string.index(",")+1:]
    img_bytes = base64.b64decode(base64string)
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    np_arr = np.array(img)
    Cvimg = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    return Cvimg

def preprocessImage(img, td=224):
    shp = img.shape[:2]  # Get height and width
    scale = min(shp) / td  # Calculate scaling factor
    new_size = tuple(int(dim * scale) for dim in shp)  # Calculate new dimensions

    # Resize and center crop the image
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    pad_h = td - img.shape[0]
    pad_w = td - img.shape[1]
    top, bottom = pad_h // 2, pad_h // 2 + pad_h % 2
    left, right = pad_w // 2, pad_w // 2 + pad_w % 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Convert to float32 and normalize to range [0, 1]
    img = img.astype(np.float32) / 255.0
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
    #psi = psi.astype(np.float32)
    # Load the TensorFlow Lite model
    interpreter = lite.Interpreter(model_path="1.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], np.expand_dims(psi, axis=0))

    # Run inference
    interpreter.invoke()

    # Get output tensor
    r = interpreter.get_tensor(output_details[0]["index"])[0]

    # post process the model output
    base64str = postprocessImage(r)
    return base64str
