from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class YOLOv7ONNXDetector:
    def __init__(self, model_path=None, conf_threshold=0.25, nms_threshold=0.45):
        base_dir = Path(__file__).resolve().parent
        self.model_path = Path(model_path) if model_path else base_dir.parent / "onnx" / "elderly.onnx"
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = ["lansia", "muda"]

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model ONNX tidak ditemukan: {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image_bytes):
        image = self._decode_image(image_bytes)
        original_height, original_width = image.shape[:2]
        input_tensor = self._preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = outputs[0][0]

        return self._postprocess(predictions, original_width, original_height)

    def _decode_image(self, image_bytes):
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("File gambar tidak bisa dibaca.")

        return image

    def _preprocess(self, image):
        img = cv2.resize(image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _postprocess(self, predictions, original_width, original_height):
        boxes = []
        scores = []
        class_ids = []

        x_scale = original_width / 640
        y_scale = original_height / 640

        for pred in predictions:
            object_conf = float(pred[4])

            if object_conf < self.conf_threshold:
                continue

            class_scores = pred[5:]
            class_id = int(np.argmax(class_scores))
            class_score = float(class_scores[class_id])
            confidence = object_conf * class_score

            if confidence < self.conf_threshold:
                continue

            x, y, w, h = pred[:4]
            x *= x_scale
            y *= y_scale
            w *= x_scale
            h *= y_scale

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)

            boxes.append([x1, y1, int(w), int(h)])
            scores.append(float(confidence))
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )

        detections = []
        for index in np.array(indices).flatten():
            x, y, w, h = boxes[int(index)]
            class_id = class_ids[int(index)]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}",
                    "confidence": round(scores[int(index)], 4),
                    "box": [x, y, w, h],
                }
            )

        return detections
