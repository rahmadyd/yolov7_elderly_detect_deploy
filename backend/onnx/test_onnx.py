import cv2
import numpy as np
import onnxruntime as ort
import os
from datetime import datetime

# ==========================
# LOAD MODEL
# ==========================

MODEL_PATH = "elderly.onnx"
IMAGE_PATH = "D:/projects/yolov7_elderly_detect_deploy/backend/onnx/testing_2young_1elderly.png"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# ==========================
# LOAD IMAGE
# ==========================

image = cv2.imread(IMAGE_PATH)

original_height, original_width = image.shape[:2]

# ==========================
# PREPROCESS
# ==========================

img = cv2.resize(image, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.astype(np.float32) / 255.0

img = np.transpose(img, (2, 0, 1))

img = np.expand_dims(img, axis=0)

# ==========================
# INFERENCE
# ==========================

outputs = session.run(
    None,
    {input_name: img}
)

predictions = outputs[0][0]

# ==========================
# POST PROCESS
# ==========================

boxes = []
scores = []
class_ids = []

CONF_THRESHOLD = 0.25

for pred in predictions:

    object_conf = pred[4]

    if object_conf < CONF_THRESHOLD:
        continue

    class_scores = pred[5:]

    class_id = np.argmax(class_scores)
    class_score = class_scores[class_id]

    confidence = object_conf * class_score

    if confidence < CONF_THRESHOLD:
        continue

    x, y, w, h = pred[:4]

    # konversi dari ukuran 640x640
    x_scale = original_width / 640
    y_scale = original_height / 640

    x *= x_scale
    y *= y_scale
    w *= x_scale
    h *= y_scale

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)

    boxes.append([x1, y1, int(w), int(h)])
    scores.append(float(confidence))
    class_ids.append(class_id)

# ==========================
# NMS
# ==========================

indices = cv2.dnn.NMSBoxes(
    boxes,
    scores,
    score_threshold=0.25,
    nms_threshold=0.45
)

# ==========================
# DRAW BOXES
# ==========================

for i in indices:

    i = int(i)

    x, y, w, h = boxes[i]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    classes = ["lansia", "muda"]

    label = f"{classes[class_ids[i]]}: {scores[i]:.2f}"

    cv2.putText(
        image,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

# ==========================
# SAVE RESULT
# ==========================

save_dir = "runs/detect"
os.makedirs(save_dir, exist_ok=True)

filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")

save_path = os.path.join(save_dir, filename)

cv2.imwrite(save_path, image)

print(f"Hasil tersimpan di: {save_path}")


# ==========================
# SHOW RESULT
# ==========================

cv2.imshow("YOLOv7 ONNX", image)

cv2.waitKey(0)
cv2.destroyAllWindows()