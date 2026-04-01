import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import os, numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import save_image
# from paddleocr import TextRecognition
# PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = True
from manga_ocr import MangaOcr
from PIL import Image, ImageDraw, ImageFont
import textwrap
import ollama

NUM_CLASSES = 3 # background + japanese + english

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("../model/best_model.pth", map_location=device))
model.to(device)
model.eval()

img_path = "../images/10.webp"

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_tensor = T.ToTensor()(img_rgb).to(device)

with torch.no_grad():
    output = model([img_tensor])[0]

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

FONT_PATH = r"../fonts/CC Wild Words Roman.ttf"

def wrap_text_pixel(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = current + " " + word if current else word
        bbox = draw.textbbox((0, 0), test, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current = test
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines

def put_eng_text(image, x1, y1, x2, y2, text):
    cv2.rectangle(image, (x1,y1),(x2,y2), (255,255,255), -1)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    # font = ImageFont.load_default()
    bubble_width = (x2-x1) -20
    bubble_height = (y2-y1) - 20

    font_size = 40

    while font_size > 10:
        font = ImageFont.truetype(FONT_PATH, font_size)
        lines = wrap_text_pixel(draw, text, font, bubble_width)
        line_height = draw.textbbox((0,0), 'Ay', font=font)[3]
        total_height = line_height * len(lines)
        if total_height <= bubble_height:
            break
        font_size = font_size - 2
    
    y_text = y1 + ((bubble_height - total_height) // 2)

    for line in lines:
        bbox = draw.textbbox((0,0),line,font=font)
        line_width = bbox[2] - bbox[0]
        x_text = x1 + ((bubble_width - line_width) // 2)
        draw.text((x_text, y_text), line, fill=(0,0,0), font=font)
        y_text += line_height

    # wrapped = textwrap.fill(text, width=width)
    # draw.multiline_text((x1+10,y1+10), wrapped, fill=(0,0,0), font=font, align="center")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


client =ollama.Client(host="http://127.0.0.1:11434")
manga_ocr = MangaOcr()
for box in box_ary:
    crop = box["crop"]
    label = box["label"]
    score = box["score"]
    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # rec_model = TextRecognition(model_name="PP-OCRv5_server_rec")
    # result = rec_model.predict(input=crop)
    
    # for res in result:
    #     print(res['rec_text'])
    #     # In 3.0, the result object stores data in these keys
    text = manga_ocr(image)
    prompt = f"""
    You are a manga dialogue translator.

    Task:
    Translate the text into natural English.

    Rules:
    - Output ONLY the English translation.
    - Change any japanese text to its proper english meaning, only if it is not a Name.
    - Do NOT explain anything.
    - Do NOT add notes.
    - Do NOT repeat the original text.
    - Keep names unchanged.
    - Preserve tone and emotion.
    - Do NOT censor or alter explicit content (violence, sexual language, insults). Translate it faithfully.
    - If the input is only symbols (… ．．． etc.), return them unchanged.

    Text:
    {text}
    """
    print(text)
    text = text.replace(" ", "")
    response = client.chat(model="qwen2.5:7b-instruct", messages=[{"role": "user", "content": prompt}])
    print(response["message"]["content"])
    img_rgb = put_eng_text(img_rgb, x1=x1, y1=y1, x2=x2, y2=y2, text=response['message']['content'])
    print("--------------------------------")

os.makedirs("../translatedImage", exist_ok=True)
cv2.imwrite("../translatedImage/manga1.png", img_rgb)