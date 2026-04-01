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

# def put_eng_text(image, x1, y1, x2, y2, text):
#     cv2.rectangle(image, (x1,y1),(x2,y2), (255,255,255), -1)
#     pil_image = Image.fromarray(image)
#     draw = ImageDraw.Draw(pil_image)
#     # font = ImageFont.load_default()
#     bubble_width = (x2-x1) -20
#     bubble_height = (y2-y1) - 20

#     font_size = 30

#     while font_size > 10:
#         font = ImageFont.truetype(FONT_PATH, font_size)
#         lines = wrap_text_pixel(draw, text, font, bubble_width)
#         line_height = draw.textbbox((0,0), 'Ay', font=font)[3]
#         total_height = line_height * len(lines)
#         if total_height <= bubble_height:
#             break
#         font_size = font_size - 2
    
#     y_text = y1 + ((bubble_height - total_height) // 2)

#     for line in lines:
#         bbox = draw.textbbox((0,0),line,font=font)
#         line_width = bbox[2] - bbox[0]
#         if x1 < 5:
#             x_text = 10 + ((bubble_width - line_width) // 2)
#         else:
#             x_text = x1 + ((bubble_width - line_width) // 2)
#         draw.text((x_text, y_text), line, fill=(0,0,0), font=font)
#         y_text += line_height

#     return np.array(pil_image)

def put_eng_text(image, x1, y1, x2, y2, text):
    cv2.rectangle(image, (x1,y1),(x2,y2), (255,255,255), -1)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    bubble_width = (x2 - x1) - 20
    bubble_height = (y2 - y1) - 20

    font_size = min((y2 - y1)//3, 28)  # better starting size

    while font_size > 10:
        font = ImageFont.truetype(FONT_PATH, font_size)

        lines = wrap_text_pixel(draw, text, font, bubble_width)

        line_height = draw.textbbox((0,0), 'Ay', font=font)[3]
        total_height = line_height * len(lines)

        # 🔥 FIX: check BOTH height + width
        fits_height = total_height <= bubble_height
        fits_width = all(
            draw.textbbox((0,0), line, font=font)[2] <= bubble_width
            for line in lines
        )

        if fits_height and fits_width:
            break

        font_size -= 2

    y_text = y1 + ((bubble_height - total_height) // 2)

    for line in lines:
        bbox = draw.textbbox((0,0), line, font=font)
        line_width = bbox[2]

        if x1 < 5:
            x_text = 10 + ((bubble_width - line_width) // 2)
        else:
            x_text = x1 + ((bubble_width - line_width) // 2)

        draw.text((x_text, y_text), line, fill=(0,0,0), font=font)
        y_text += line_height

    return np.array(pil_image)
    # wrapped = textwrap.fill(text, width=width)
    # draw.multiline_text((x1+10,y1+10), wrapped, fill=(0,0,0), font=font, align="center")
    # return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

system_prompt = f"""
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
            """
if __name__ == "__main__":
    # geting the images from the folder
    directory_path = "/home/zumbie/Downloads/HENTAI/[GREAT芥 (tokyo)] ホルンの魔女つかまえた (メロンブックス デジタルver) (アークザラッド)"
    # directory_path ="/home/zumbie/Downloads/[Tamaya Gekijou (Tamaya Cinema)] Zetsurin Oji-san to Osananajimi no Musume no, Sukebe na Suujitsukan Day 2 [Chinese]"
    
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

            text = manga_ocr(image)
        
            text = text.replace(" ", "")
            response = client.chat( model="qwen2.5:7b-instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ]
                )
            # response = client.chat(model="qwen2.5:7b-instruct", messages=[{"role": "user", "content": prompt}])
            # print(response["message"]["content"])
            img_rgb = put_eng_text(img_rgb, x1=x1, y1=y1, x2=x2, y2=y2, text=response['message']['content'])

        # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        pillow_image = Image.fromarray(img_rgb)
        pillow_image.save(os.path.join(save_images_path,f"{i}.png"))
        # cv2.imwrite(os.path.join(save_images_path,f"{i}.png"), img_rgb)
        translated_images.append(os.path.join(save_images_path,f"{i}.png"))
        print("Page :", i+1, " Completed")
    print(translated_images)
    with open(os.path.join(directory_path, "translated_images.pdf"), "wb") as f: 
        f.write(img2pdf.convert(translated_images))
    f.close()
    shutil.rmtree(save_images_path)