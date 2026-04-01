import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import os
from matplotlib import pyplot as plt
from torchvision.utils import save_image

NUM_CLASSES = 3 # background + japanese + english

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("./model/best_model.pth", map_location=device))
model.to(device)
model.eval()

img_path = "./images/manga2.jpg"

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_tensor = T.ToTensor()(img_rgb).to(device)

with torch.no_grad():
    output = model([img_tensor])[0]

boxes = output["boxes"].cpu().numpy()
scores = output["scores"].cpu().numpy()
labels = output["labels"].cpu().numpy()

SCORE_THRESH = 0.55

for box, score, label in zip(boxes, scores, labels):
    if score < SCORE_THRESH:
        continue

    x1, y1, x2, y2 = map(int, box)
    
    color = (0, 255, 0) if label == 1 else (255, 0,0)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img_bgr, f"{label} {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

plt.imshow(img_bgr)
plt.title("ComLook")
plt.axis("off")
plt.show()
plt.close()