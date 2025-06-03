import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# === Path model dan gambar ===
model_path = "yolov5s.onnx"
image_path = "image.png"
coco_names_path = "coco.names"

# === Parameter deteksi ===
input_size = 640
conf_threshold = 0.4
nms_threshold = 0.5

# === Load class labels COCO ===
if os.path.exists(coco_names_path):
    with open(coco_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = None

tosca = (208, 224, 64)

def get_color(_): return tosca

# === Load gambar ===
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Gagal membaca gambar: {image_path}")
orig_h, orig_w = image.shape[:2]

# === Preprocessing ===
input_image = cv2.resize(image, (input_size, input_size))
input_tensor = input_image.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# === Load model ONNX ===
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# === Mulai timer ===
start_time = time.time()

# === Inference ===
output = session.run(None, {input_name: input_tensor})[0][0]  # shape: (N, 85)

# === Filter prediksi confidence
conf_mask = output[:, 4] > conf_threshold
output = output[conf_mask]

if output.shape[0] > 0:
    scores = output[:, 4] * output[:, 5:].max(axis=1)
    score_mask = scores > conf_threshold
    output = output[score_mask]
    scores = scores[score_mask]
    class_ids = output[:, 5:].argmax(axis=1)

    cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    x = ((cx - w / 2) * orig_w / input_size).astype(int)
    y = ((cy - h / 2) * orig_h / input_size).astype(int)
    w = (w * orig_w / input_size).astype(int)
    h = (h * orig_h / input_size).astype(int)

    boxes = np.stack([x, y, w, h], axis=1)

    # === NMS pakai OpenCV (paling cepat walau kamu gak pake GPU)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold)

    # === Stop timer
    end_time = time.time()
    print(f"Deteksi selesai dalam {end_time - start_time:.3f} detik.")

    # === Gambar hasil
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        label_text = class_names[class_ids[i]] if class_names else f"Class {class_ids[i]}"
        label = f"{label_text}: {scores[i]:.2f}"

        cv2.rectangle(image, (x, y), (x + w, y + h), tosca, 1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tosca, 1)
else:
    end_time = time.time()
    print("Tidak ada objek terdeteksi.")
    print(f"Waktu: {end_time - start_time:.3f} detik.")

# === Tampilkan hasil
cv2.imshow("YOLOv5 Fast CPU", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
