import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import os, numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from manga_ocr import MangaOcr
from PIL import Image, ImageDraw, ImageFont
import textwrap
import ollama
import img2pdf
from tqdm import tqdm
import shutil

# def wrap_text_pixel(draw, text, font, max_width):
#     words = text.split()
#     lines = []
#     current = ""
#     for word in words:
#         bbox = draw.textbbox((0, 0), word, font=font)
#         word_width = bbox[2] - bbox[0] # Correct width calculation
        
#         if word_width > max_width:
#             if current: lines.append(current)
#             # lines.append(word)
#             # current = ""
#             # continue
#             #------CHOPPING THE WORD INTO SMALLER PIECES------
#             temp_word = word
#             while True:
#                 split_idx = 0
#                 for i in range(1, len(temp_word)):
#                     test_part = temp_word[:i] + "-"
#                     part_w = draw.textbbox((0,0), test_part, font=font)[2] - draw.textbbox((0,0), test_part, font=font)[0]
#                     if part_w > max_width:
#                         break
#                     split_idx = i
#                  # If we can't even fit one char + '-', just force split at 1 char
#                 split_idx = max(1, split_idx)
                
#                 # Add the chopped part to lines
#                 lines.append(temp_word[:split_idx] + "-")
#                 temp_word = temp_word[split_idx:]
                
#                 # Check remaining part
#                 rem_w = draw.textbbox((0, 0), temp_word, font=font)[2] - draw.textbbox((0, 0), temp_word, font=font)[0]
#                 if rem_w <= max_width:
#                     current = temp_word # Remaining bit becomes the start of the next line
#                     break
#             continue
#             #------CHOPPING THE WORD INTO SMALLER PIECES------

#         test = current + " " + word if current else word
#         test_bbox = draw.textbbox((0, 0), test, font=font)
#         test_width = test_bbox[2] - test_bbox[0] # Correct width calculation
        
#         if test_width <= max_width:
#             current = test
#         else:
#             lines.append(current)
#             current = word
            
#     if current: lines.append(current)
#     return lines

# def put_all_eng_text(image, panel_boxes):
#     # Quick NumPy white-out
#     for x1, y1, x2, y2, _ in panel_boxes:
#         image[y1:y2, x1:x2] = 255 
        
#     pil_img = Image.fromarray(image)
#     draw = ImageDraw.Draw(pil_img)

#     for x1, y1, x2, y2, text in panel_boxes:
#         # Use a slightly larger padding (e.g., 15) to ensure text doesn't touch edges
#         padding = 15
#         w, h = (x2 - x1) - padding, (y2 - y1) - padding
        
#         low, high = 8, 24
#         final_font, final_lines, final_total_h = None, [], 0
        
#         while low <= high:
#             mid = (low + high) // 2
#             f = ImageFont.truetype(FONT_PATH, mid)
#             lines = wrap_text_pixel(draw, text, f, w)
            
#             line_metrics = draw.textbbox((0, 0), "Ay", font=f)
#             line_h = (line_metrics[3] - line_metrics[1])*1.2
#             total_h = line_h * len(lines)
            
#             if total_h <= h:
#                 final_font, final_lines, final_total_h = f, lines, total_h
#                 low = mid + 1
#             else:
#                 high = mid - 1
        
#         if final_font:
#             line_metrics = draw.textbbox((0, 0), "Ay", font=final_font)
#             line_h = line_metrics[3] - line_metrics[1]
            
#             # Start Y: Center the block vertically
#             start_y = y1 + ( (y2 - y1) - final_total_h ) // 2
            
#             for line in final_lines:
#                 l_bbox = draw.textbbox((0, 0), line, font=final_font)
#                 l_w = l_bbox[2] - l_bbox[0] # Correct width
                
#                 # Start X: Center this specific line horizontally
#                 start_x = x1 + ( (x2 - x1) - l_w ) // 2
                
#                 draw.text((start_x, start_y), line, fill=0, font=final_font)
#                 start_y += line_h

#     return np.array(pil_img)
def put_all_eng_text(image, panel_boxes):
    for x1, y1, x2, y2, _ in panel_boxes:
        image[y1:y2, x1:x2] = 255

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    font_size =  17  # ✅ constant size
    line_spacing = 5 # ✅ vertical gap between lines
    font = ImageFont.truetype(FONT_PATH, font_size)

    for x1, y1, x2, y2, text in panel_boxes:
        padding = 12
        max_width = (x2 - x1) - padding * 2

        lines = wrap_text_pixel(draw, text, font, max_width)

        bbox = draw.textbbox((0, 0), "Ay", font=font)
        line_height = (bbox[3] - bbox[1]) + line_spacing

        total_height = line_height * len(lines)

        start_y = y1 + max(5, ((y2 - y1) - total_height) // 2)

        for line in lines:
            lb = draw.textbbox((0, 0), line, font=font)
            line_width = lb[2] - lb[0]

            start_x = x1 + max(5, ((x2 - x1) - line_width) // 2)

            draw.text((start_x, start_y), line, fill=0, font=font)

            start_y += line_height

    return np.array(pil_img)

def wrap_text_pixel(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()

        bbox = draw.textbbox((0, 0), test, font=font)
        test_width = bbox[2] - bbox[0]

        if test_width <= max_width:
            current = test
            continue

        # current line full → push it
        if current:
            lines.append(current)

        # if single word itself too large, chop with hyphen
        bbox_word = draw.textbbox((0, 0), word, font=font)
        if bbox_word[2] - bbox_word[0] > max_width:
            temp = ""
            for ch in word:
                trial = temp + ch
                bw = draw.textbbox((0, 0), trial + "-", font=font)[2]
                if bw > max_width:
                    lines.append(temp + "-")
                    temp = ch
                else:
                    temp = trial
            current = temp
        else:
            current = word

    if current:
        lines.append(current)

    return lines


system_prompt = f"""
            You are a adult manga dialogue translator.

            Task:
            Translate the texts into natural English. keep the order of the text.

            Rules:
            - Output ONLY the English translations, if some are noises (sexual, insults, etc.) then spell the noises in english letters.
            - Understand the context by the all the texts and translate like real conversation.
            - Change any japanese text to its proper english meaning, only if it is not a noun.
            - Do NOT explain anything.
            - Do NOT add notes.
            - Do NOT repeat the original texts.
            - Keep names unchanged.
            - Preserve tone and emotion.
            - Do NOT censor or alter explicit contents (violence, sexual language, insults). Translate it faithfully.
            - If the input is only symbols (… ．．． etc.), return them unchanged.
            - If anytext contains mother or father or synonym of it then change it to aunt or uncle. Same for Son but change it to nephew.
            """
if __name__ == "__main__":
    # geting the images from the folder
    directory_path = "/home/zumbie/Downloads/HENTAI/(成年コミック) [ジョン・K・ペー太] マン・コンプリート [DL版]"

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
    for i in images:
        os.rename(os.path.join(directory_path, i), os.path.join(directory_path, i.replace("b182asnw02346_", "")))
    images = os.listdir(directory_path)
    images=sorted(images)
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

        response = client.chat( model="qwen2.5:7b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": joined_text}
                ]
            )
        lines = response['message']['content'].strip().split("\n")
        translations = {}
        index = 0
        for line in lines:
            if len(line.split(":", 1)) == 1:
                continue
            idx, txt = line.split(":", 1)
            try:
                translations[int(idx)] = txt
                index = int(idx)
            except:
                print("Error in line : ",line)
                translations[index+1] = txt
                index += 1
        index = 0
        panel_boxes = []
        for j, box in enumerate(box_ary):
            panel_boxes.append([box['x1'], box['y1'], box['x2'], box['y2'], translations.get(j, "")])
        img_rgb = put_all_eng_text(img_rgb, panel_boxes)

        pillow_image = Image.fromarray(img_rgb)
        pillow_image.save(os.path.join(save_images_path,f"{i}.png"))
        translated_images.append(os.path.join(save_images_path,f"{i}.png"))
        print("Page :", i+1, " Completed")
    with open(os.path.join(directory_path, "translated_images.pdf"), "wb") as f: 
        f.write(img2pdf.convert(translated_images))
    f.close()
    shutil.rmtree(save_images_path)