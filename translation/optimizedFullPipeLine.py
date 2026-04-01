import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import os, numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import save_image
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from manga_ocr import MangaOcr
from PIL import Image, ImageDraw, ImageFont
import textwrap
import ollama
import img2pdf
from tqdm import tqdm
import shutil

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
def put_all_eng_text(image, panel_boxes):
    for i in panel_boxes:
        x1, y1, x2, y2, _ = i
        cv2.rectangle(image, (x1+5, y1+5), (x2-5, y2-5), (255,255,255), -1)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for i in panel_boxes:
        x1, y1, x2, y2, text = i
        bubble_width = (x2-x1) -20
        bubble_height = (y2-y1) - 20
        # font_size = min((y2 - y1)//3, 28)  # better starting size
        font_size = 35
        while font_size > 10:
            font = ImageFont.truetype(FONT_PATH, font_size)
            lines = wrap_text_pixel(draw, text, font, bubble_width)
            line_height = draw.textbbox((0, 0), 'Ay', font=font)[3]
            total_height = line_height * len(lines)
            if total_height <= bubble_height:
                break
            font_size = font_size - 2
        y_text = y1 + ((bubble_height - total_height) // 2)
        for line in lines:
            bbox = draw.textbbox((0,0),line,font=font)
            line_width = bbox[2] - bbox[0]
            if x1 < 5:
                x_text = 10 + ((bubble_width - line_width) // 2)
            else:
                x_text = x1 + ((bubble_width - line_width) // 2)
            draw.text((x_text, y_text), line, fill=(0,0,0), font=font)
            y_text += line_height
    return np.array(pil_image)

def put_eng_text(image, x1, y1, x2, y2, text):
    cv2.rectangle(image, (x1,y1),(x2,y2), (255,255,255), -1)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    # font = ImageFont.load_default()
    bubble_width = (x2-x1) -20
    bubble_height = (y2-y1) - 20

    font_size = 35

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
        if x1 < 5:
            x_text = 10 + ((bubble_width - line_width) // 2)
        else:
            x_text = x1 + ((bubble_width - line_width) // 2)
        draw.text((x_text, y_text), line, fill=(0,0,0), font=font)
        y_text += line_height

    return np.array(pil_image)

system_prompt = f"""
            You are a adult manga dialogue translator.

            Task:
            Translate the texts into natural English. keep the order of the text.

            Rules:
            - Output ONLY the English translations.
            - Understand the context by the all the texts and translate like real conversation.
            - Change any japanese text to its proper english meaning, only if it is not a noun.
            - Do NOT explain anything.
            - Do NOT add notes.
            - Do NOT repeat the original texts.
            - Keep names unchanged.
            - Preserve tone and emotion.
            - Do NOT censor or alter explicit contents (violence, sexual language, insults). Translate it faithfully.
            - If the input is only symbols (… ．．． etc.), return them unchanged.
            """
if __name__ == "__main__":
    # geting the images from the folder
    directory_path = "../manga"

    FONT_PATH = r"../fonts/CC Wild Words Roman.ttf"

    client =ollama.Client(host="http://127.0.0.1:11434")
    manga_ocr = MangaOcr()
    NUM_CLASSES = 3 # background + japanese + english

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("../model/best_model.pth", map_location=device))
    model.to(device)
    model.eval()


    save_images_path = os.path.join("../", "MangaTranslatedImages")
    os.makedirs(save_images_path, exist_ok=True)
    images = os.listdir(directory_path)
    images=sorted(images)
    print(images)
    translated_images = []
    for image_path, i in tqdm(zip(images, range(len(images)))):
        if image_path.endswith(".pdf"):
            continue

        img_path = os.path.join(directory_path, image_path)

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

        # Optimizations starts here
        texts = []
        for box in box_ary:
            crop = box["crop"]
            label = box["label"]
            score = box["score"]
            image = Image.fromarray(crop)

            text = manga_ocr(image).replace(" ", "")
            texts.append(text)
        
        joined_text = "\n".join([f"{i}:{text}" for i ,text in enumerate(texts)])
        print(joined_text)

        response = client.chat( model="qwen2.5:7b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": joined_text}
                ]
            )
        lines = response['message']['content'].strip().split("\n")
        translations = {}
        for line in lines:
            if len(line.split(":", 1)) == 1:
                continue
            idx, txt = line.split(":", 1)
            translations[int(idx)] = txt
        print(translations)
        panel_boxes = []
        for i, box in enumerate(box_ary):
            panel_boxes.append([box['x1'], box['y1'], box['x2'], box['y2'], translations.get(i, "")])
        img_rgb = put_all_eng_text(img_rgb, panel_boxes)

        pillow_image = Image.fromarray(img_rgb)
        pillow_image.save(os.path.join(save_images_path,f"{i}.png"))
        translated_images.append(os.path.join(save_images_path,f"{i}.png"))
        print("Page :", i+1, " Completed")
    with open(os.path.join(directory_path, "translated_images.pdf"), "wb") as f: 
        f.write(img2pdf.convert(translated_images))
    f.close()
    shutil.rmtree(save_images_path)