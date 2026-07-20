import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import re
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
from datetime import datetime

# directory_path = "/home/zumbie/Downloads/HENTAI/(成年コミック) [ジョン・K・ペー太] マン・コンプリート [DL版]"
# directory_path = "/home/zumbie/Downloads/HENTAI/(C106) [STUDIO VANGUARD (TWILIGHT)] V250817 [Chinese]"
directory_path = r"/home/zumbie/Downloads/HENTAI/nhentai-665588 - [ma_shika (A_shika)] Tenshi Hirotta kara Haramaseru ~Ojii-san Senyou Botebara Onaho ni Naru made no"
# directory_path = "/home/zumbie/Downloads/HENTAI/testfuck"


def wrap_text_pixel(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        bbox = draw.textbbox((0, 0), word, font=font)
        word_width = bbox[2] - bbox[0] # Correct width calculation
        
        if word_width > max_width:
            if current: lines.append(current)
            # lines.append(word)
            # current = ""
            # continue
            #------CHOPPING THE WORD INTO SMALLER PIECES------
            temp_word = word
            while True:
                split_idx = 0
                for i in range(1, len(temp_word)):
                    test_part = temp_word[:i] + "-"
                    part_w = draw.textbbox((0,0), test_part, font=font)[2] - draw.textbbox((0,0), test_part, font=font)[0]
                    if part_w > max_width:
                        break
                    split_idx = i
                 # If we can't even fit one char + '-', just force split at 1 char
                split_idx = max(1, split_idx)
                
                # Add the chopped part to lines
                lines.append(temp_word[:split_idx] + "-")
                temp_word = temp_word[split_idx:]
                
                # Check remaining part
                rem_w = draw.textbbox((0, 0), temp_word, font=font)[2] - draw.textbbox((0, 0), temp_word, font=font)[0]
                if rem_w <= max_width:
                    current = temp_word # Remaining bit becomes the start of the next line
                    break
            continue
            #------CHOPPING THE WORD INTO SMALLER PIECES------

        test = current + " " + word if current else word
        test_bbox = draw.textbbox((0, 0), test, font=font)
        test_width = test_bbox[2] - test_bbox[0] # Correct width calculation
        
        if test_width <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
            
    if current: lines.append(current)
    return lines

def put_all_eng_text(image, panel_boxes):
    line_spacing = 1.4
    # Quick NumPy white-out
    for x1, y1, x2, y2, _ in panel_boxes:
        image[y1:y2, x1:x2] = 255  
        # image[y1+5:y2-5, x1+5:x2-5] = 255 

        
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    for x1, y1, x2, y2, text in panel_boxes:
        # Use a slightly larger padding (e.g., 15) to ensure text doesn't touch edges
        padding = 8
        w, h = (x2 - x1) - padding, (y2 - y1) - padding+8
        
        low, high = 14, 22#30
        final_font, final_lines, final_total_h = None, [], 0
        
        while low <= high:
            mid = (low + high) // 2
            f = ImageFont.truetype(FONT_PATH, mid)
            lines = wrap_text_pixel(draw, text, f, w)
            
            line_metrics = draw.textbbox((0, 0), "Ay", font=f)
            line_h = (line_metrics[3] - line_metrics[1])*line_spacing   #1.2
            total_h = line_h * len(lines)
            
            if total_h <= h:
                final_font, final_lines, final_total_h = f, lines, total_h
                low = mid + 1
            else:
                high = mid - 1
        
        if final_font:
            line_metrics = draw.textbbox((0, 0), "Ay", font=final_font)
            line_h = (line_metrics[3] - line_metrics[1])*line_spacing
            
            # Start Y: Center the block vertically
            start_y = y1 + ( (y2 - y1) - final_total_h ) // 2
            
            for line in final_lines:
                l_bbox = draw.textbbox((0, 0), line, font=final_font)
                l_w = l_bbox[2] - l_bbox[0] # Correct width
                
                # Start X: Center this specific line horizontally
                start_x = x1 + ( (x2 - x1) - l_w ) // 2
                
                draw.text((start_x, start_y), line, fill=0, font=final_font)
                start_y += line_h

    return np.array(pil_img)


def get_system_prompt(text):

    prompt = f"""
    You are a literal translation engine.

    Translate Japanese text to natural English with the essence and tone of the original.

    Rules:
    - Do not add any extra line or any kind of explanation. only give Bubble (number): "meaning".
    - Strictly Follow the example output format with proper 'Bubble (number)' and without extra white spaces.
    - Never miss any bubble. Translate all of them in the same order.
    - Preserve insults, explicit wording and noises.
    - Understand the context by the all the texts and translate like real conversation.
    - Keep the same numbering and DO NOT MISS ANY BUBBLE.
    - Preserve moans, screams, and vulgar sounds naturally in English.
    - Output ONLY translated lines and do not add any explanations.
    - Do not add extra bubbles.


    EXAMPLE INPUT:
        Bubble 0: こんにちは
        Bubble 1: ばか
    EXAMPLE OUTPUT:
        Bubble 0: Hello
        Bubble 1: Idiot

    Input:
    {text}

    Output:
        """
    return prompt

    system_prompt = f"""
                You are a adult manga dialogue translator.

                TASK:
                - Translate the texts into natural English with the essence and tone of the original.

                CONTEXT:
                - The conversations are between charachters.
                - Conversations are spicy, vulgar and explicit.

                RULES:
                - Understand the context by the all the texts and translate like real conversation.
                - Consider the pronouns FEMALE if no proper context in bubble to decide the gender of the character.
                - Change any japanese/Chinese text to its proper english meaning, only if it is not a noun.
                - Keep names of places unchanged.
                - Preserve tone and emotion.
                - Do NOT censor or alter explicit contents (violence, sexual language, insults). Translate it faithfully.
                - Out put names of body parts like "penis", "vagina", "anus" etc faithfully.
                - Preserve moans, screams, and vulgar sounds naturally in English.
                
                OUTPUT FORMAT:
                - Do NOT explain anything.
    ↓           - Do NOT add notes.
                - Return exactly TRANSLATION line per bubble with correct bubble order.       
                - EXAMPLE INPUT:
                Bubble 0: こんにちは
                Bubble 1: ばか
                - EXAMPLE OUTPUT:
                Bubble 0: Hello
                Bubble 1: Idiot

                FAILURE CONDITIONS (DO NOT DO THESE):
                - Missing "Bubble"
                - Adding explanations
                - Changing numbering
                - Adding extra lines

                Below is the text to translate:
                {text}
                """
    return system_prompt    
            # - Use strict output format:
            #     Bubble 0: ...
            #     Bubble 1: ...
def incestkiller(txt):
    txt = re.sub(r"\bmother\b", "ane sama", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bmom\b", "ane sama", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bmomma\b", "ane sama", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bfather\b", "ojisan", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bdad\b", "ojisan", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bson\b", "boya", txt, flags=re.IGNORECASE)
    return txt
    if " mother " in txt.lower():
        txt = txt.replace(" mother ", " ane sama ")
    if  " mom " in txt.lower():
        txt = txt.replace(" mom ", " ane sama ")
    if " momma " in txt.lower():
        txt = txt.replace(" momma ", " ane sama ")
    if " father " in txt.lower():
        txt = txt.replace(" father ", " ojisan ")
    if " dad " in txt.lower():
        txt = txt.replace(" dad ", " ojisan ")
    if " son " in txt.lower():
        txt = txt.replace(" son ", " boya ")
    return txt

if __name__ == "__main__":
    # geting the images from the folder

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
    # images = os.listdir(directory_path)
    # for i in images:
    #     os.rename(os.path.join(directory_path, i), os.path.join(directory_path, i.replace("pg", "")))
    images = os.listdir(directory_path)
    images=sorted(images)
    translated_images = []
    for image_path, i in tqdm(zip(images, range(len(images)))):
        if image_path.endswith(".pdf" or ".txt"):
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
        
        if len(box_ary)>0:
            
            joined_text = "\n".join([f"Bubble {i}: {text}" for i ,text in enumerate(texts)])
            # print(joined_text)
            # response = client.chat( model="huihui_ai/phi4-mini-abliterated", #"phi4-mini:latest",
            #         messages=[
            #             {"role": "system", "content": system_prompt},
            #             {"role": "user", "content": joined_text}
            #         ]
            #     )
            print("this is joined text : ",joined_text)
            system_prompt = get_system_prompt(joined_text)
            response  = client.generate(
                # model='huihui_ai/phi4-mini-abliterated',
                model="richardyoung/qwen2.5-7b-instruct-abliterated",
                prompt= system_prompt,
                options={
                    "temperature":0.0,
                    "top_p": 0.9,
                    "num_predict": 256,
                    "repeat_penalty": 1.2,
                    "stop": [f"Bubble {len(texts)}:"],
                    }
            )
            print("this is response : ",response["response"])
            lines = response['response'].strip().split("\n")
            # lines = response['message']['content'].strip().split("\n")

            translations = {}
            index = -1
            for line in lines:
                index = 0
                try:
                    if len(line.split(":", 1)) == 1:
                        continue
                    idx, txt = line.split(":", 1)
                    idx = idx.replace("Bubble", "")
                    pussy = idx
                    # print(idx)
                    idx = int(idx.strip())
                except Exception as e:
                    print("IDX error : ", pussy)
                    idx = index + 1
                try:
                    # print("before incestkiller : ", txt)
                    txt = incestkiller(txt)
                    # print("after incestkiller : ", txt)
                    translations[idx] = txt
                    index = idx
                except:
                    print("Error in line : ",line, '\n...........................................................\n')
                    txt = incestkiller(txt)
                    translations[index+1] = txt
                    index += 1
            index = 0
            # print(joined_text)
            # print(translations, "\n\n\n\n\n\n")
            panel_boxes = []
            for j, box in enumerate(box_ary):
                panel_boxes.append([box['x1'], box['y1'], box['x2'], box['y2'], translations.get(j, "")])
            img_rgb = put_all_eng_text(img_rgb, panel_boxes)

        pillow_image = Image.fromarray(img_rgb)
        pillow_image.save(os.path.join(save_images_path,f"{i}.png"))
        translated_images.append(os.path.join(save_images_path,f"{i}.png"))
        print("Page :", i+1, " Completed")
    name = directory_path.split('/')[-1]
    with open(os.path.join(directory_path, f"{name} {datetime.now()}.pdf"), "wb") as f: 
        f.write(img2pdf.convert(translated_images))
    f.close()
    shutil.rmtree(save_images_path)