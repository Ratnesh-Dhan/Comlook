import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from manga_ocr import MangaOcr
from supports.ReWriter import ReWriter
from supports.Translators import Translator
from PIL import Image

class Ml:
    def __init__(self):
        NUM_CLASSES = 3 # Background + Japanese/Chinese + English
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load("../model/best_model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.manga_ocr = MangaOcr()
        # self.rewriter = ReWriter()
        self.translator = Translator()

    def bboxes(self, img_rgb):
        img_tensor = T.ToTensor()(img_rgb).to(self.device)

        with torch.no_grad():
            output = self.model([img_tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        SCORE_THRESH = 0.55

        box_ary = []
        for box, score, label in zip(boxes, scores, labels):
            if score < SCORE_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop = img_rgb[y1:y2, x1:x2]
            box_ary.append({"crop": crop, "label": label, "score": score, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        
        panel_boxes = []
        for box in box_ary:
            crop = box["crop"]
            label = box["label"]
            score = box["score"] 
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]
            # image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(crop)

            text = self.manga_ocr(image)
        
            # text = text.replace(" ", "")
            translation, text = self.translator.translate_nllb(text)
            print("Actual Text: ", text)
            print("Raw Translation: ", translation, "\n")
            # translation = self.rewriter.rewrite_dialogue(translation)
            # print("Modified Translation: ", translation)
            panel_boxes.append([x1, y1, x2, y2, translation])

        return panel_boxes

