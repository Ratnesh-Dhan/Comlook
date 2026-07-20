import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
import shutil
import re

class Textworker:
    def __init__(self):
        self.FONT_PATH = r"../fonts/CC Wild Words Roman.ttf"
        self.save_images_path = "MangaTranslatedImages"
        os.makedirs(self.save_images_path, exist_ok=True)

    def wrap_text_pixel(self, draw, text, font, max_width):
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

    def put_all_eng_text(self, image, panel_boxes):
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
                f = ImageFont.truetype(self.FONT_PATH, mid)
                lines = self.wrap_text_pixel(draw, text, f, w)
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
        
        return pil_img
    
    def incestkiller(self, txt):
        txt = re.sub(r"\bmother\b", "ane sama", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\bmom\b", "ane sama", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\bmomma\b", "ane sama", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\bfather\b", "ojisan", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\bdad\b", "ojisan", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\bson\b", "boya", txt, flags=re.IGNORECASE)
        return txt
    def translations(self, lines):
        translations = {}
        index = -1
        for line in lines:
            index = 0
            try:
                if len(line.split(":", 1)) == 1:
                    continue
                idx, txt = line.split(":", 1)
                idx = idx.replace("Bubble", "")
                er = idx
                # print(idx)
                idx = int(idx.strip())
            except Exception as e:
                print("IDX error : ", er)
                print(txt)
                idx = index + 1
            try:
                txt = self.incestkiller(txt)
                translations[idx] = txt
                index = idx
            except:
                print("Error in line : ",line, '\n...........................................................\n')
                txt = self.incestkiller(txt)
                translations[index+1] = txt
                index += 1
            
        print("TRANSLATIONS : ",translations)
        return translations
    
    def get_completed_image(self, box_ary, img_rgb, translations, i):
        try:
            panel_boxes = []
            for j, box in enumerate(box_ary):
                panel_boxes.append([box['x1'], box['y1'], box['x2'], box['y2'], translations.get(j, "")])
            pillow_image = self.put_all_eng_text(img_rgb, panel_boxes)
            pillow_image.save(os.path.join(self.save_images_path,f"{i}.png"))
            return True
        except Exception as e:
            print("Exception during saving completed image ==%%==>")
            print(e)
            return 
    
    def save_final_pdf(self, directory_path):
        try:
            name = directory_path.split('/')[-1]

            # Get image files sorted numerically
            image_files = sorted(
                [f for f in os.listdir(self.save_images_path) if f.endswith(".png")],
                key=lambda x: int(os.path.splitext(x)[0])
            )

            # Open images
            images = [Image.open(os.path.join(self.save_images_path, f)).convert("RGB") for f in image_files]


            # Save as PDF
            output_pdf = os.path.join(directory_path, f"{name} {datetime.now()}.pdf")

            if images:
                images[0].save(
                    output_pdf,
                    save_all=True,
                    append_images=images[1:]
                )
            shutil.rmtree(self.save_images_path)
            
            return "Saved successfully"
        except Exception as e:
            print("Exception during saving completed PDF ==%%==>")
            print(e)
            return "Something fucked up during saving final PDF"