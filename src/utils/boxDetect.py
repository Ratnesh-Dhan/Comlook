import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import numpy as np
from PIL import Image
from manga_ocr import MangaOcr
import cv2

class BoxDetect:
    def __init__(self):
        
        NUM_CLASSES = 3 # background + japanese + english
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load("../model/best_model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.SCORE_THRESH = 0.55
        self.manga_ocr = MangaOcr()


    def get_boxes(self, img_path):
        
        # if image_path.endswith(".pdf" or ".txt"):
        #     continue

        # img_path = os.path.join(directory_path, image_path)

        # img_bgr = cv2.imread(img_path)
        # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_tensor = T.ToTensor()(img_rgb).to(self.device)

        with torch.no_grad():
            output =self.model([img_tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()


        box_ary = []

        for box, score, label in zip(boxes, scores, labels):
            if score < self.SCORE_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop = img_rgb[y1:y2, x1:x2]
            box_ary.append({"crop": crop, "label": label, "score": score, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            
        return box_ary, img_rgb

    def get_texts(self, box_ary):
        texts = []
        for box in box_ary:
            crop = box["crop"]
            image = Image.fromarray(crop)

            text = self.manga_ocr(image).replace(" ", "")
            texts.append(text)
        
        return texts