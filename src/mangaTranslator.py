import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from utils.boxDetect import BoxDetect
from utils.utilOllama import Utilollama
from utils.support import Textworker
from utils.custom_prompt import Prompts
from tqdm import tqdm
import shutil

if __name__ == "__main__":
    directory_path = r"/home/zumbie/Downloads/HENTAI/[Hasebe Souutsu] Izayoi Sakuya to Higyaku Shounentachi Soushuuhen"
    
    boxDetect = BoxDetect()
    custom_prompts = Prompts()
    olama = Utilollama(custom_prompts)
    textWorker = Textworker()

    images = os.listdir(directory_path)
    print("Total Images: ",len(images))
    print(images)
    images=sorted(images)
    print(images)
    # translated_images = []
    for image_path, i in tqdm(zip(images, range(len(images)))):
        if image_path.endswith(".pdf" or ".txt"):
            continue
        box_ary, img_rgb = boxDetect.get_boxes(os.path.join(directory_path, image_path))
        # Optimizations starts here
        texts = boxDetect.get_texts(box_ary=box_ary)
        
        if len(box_ary)>0:
            
            # print(joined_text)
            lines = olama.ollama_generate(texts)

            translations = textWorker.translations(lines)

            success_message = textWorker.get_completed_image(box_ary, img_rgb, translations, i)
            if success_message:
                print("Page :", i+1, " Completed")
            else:
                print(f"Error while saving page: {i+1}")

    print(textWorker.save_final_pdf(directory_path))