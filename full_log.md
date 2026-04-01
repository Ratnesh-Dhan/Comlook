Manga Translator – End-to-End Pipeline
Overview

This project implements an automatic manga translation pipeline that detects Japanese/Chinese text in manga pages, extracts the dialogue, translates it into English using an LLM, and reinserts the translated text back into the image.

The system combines computer vision, OCR, and large language models to build a complete manga translation workflow.

System Architecture
Manga Image
↓
Text Detection (Faster R-CNN)
↓
Text Region Cropping
↓
OCR (MangaOCR)
↓
LLM Translation (Qwen2.5)
↓
Bubble-aware Text Layout
↓
Translated Manga Page
1. Text Detection
Model

Faster R-CNN using:

torchvision.models.detection.fasterrcnn_resnet50_fpn
Dataset

Custom manga text dataset.

Split	Images
Train	158
Validation	69

Annotations format:

Pascal VOC XML

Example:

<object>
    <name>japanese</name>
    <bndbox>
        <xmin>96</xmin>
        <ymin>1381</ymin>
        <xmax>310</xmax>
        <ymax>1714</ymax>
    </bndbox>
</object>
Training Strategy

Pretrained backbone

Freeze backbone for first epochs

Early stopping based on validation loss

Small batch training

Result:
The model generalized well on unseen manga pages.

2. Dataset Issues and Fixes
Problem

Training crashed with:

AssertionError: Expected target boxes to be tensor [N,4]
Cause

Images contained <object> tags but labels were not "japanese".

This resulted in empty bounding boxes.

Solution

Filtered dataset during initialization to only include images containing valid "japanese" labels.

3. OCR Extraction
Model Used
manga-ocr (kha-white/manga-ocr-base)

Reason:
General OCR models performed poorly on manga fonts.

MangaOCR is trained on manga-style typography and vertical Japanese text.

Issue Encountered
ValueError: img_or_path must be PIL.Image
Cause

OCR model requires PIL images, not NumPy arrays.

Fix
crop_pil = Image.fromarray(crop_rgb)
4. Translation
Model
Qwen2.5-7B-Instruct
Runtime
Ollama

This allows local LLM inference.

Translation Prompt
You are a manga dialogue translator.

Rules:
- Output ONLY the English translation.
- Do NOT explain anything.
- Preserve emotion and tone.
- Do NOT censor explicit content.

Text:
{text}

English:
5. WSL + Ollama Networking Issue
Problem

WSL could not connect to Ollama running on Windows.

Solution

Used Windows host IP from:

/etc/resolv.conf

Connected using:

http://WINDOWS_IP:11434
6. LLM Output Issues
Problem

LLM generated explanations such as:

"Pirra!! translates to..."
Solution

Used strict prompt constraints and post-processing:

translation = output.split("\n")[0]
7. Text Rendering

Goal: Insert translated English back into manga bubbles.

Steps:

Remove original Japanese text

Wrap English text

Fit text inside bubble

Render using manga font

Font used:

CC Wild Words Roman
8. Text Wrapping Problem
Problem

Text appeared on a single line.

Cause

textwrap.fill() uses character count, not pixel width.

Fix

Implemented pixel-based text wrapping using:

draw.textbbox()
9. Bubble-aware Layout

Implemented layout system that:

wraps text by pixel width

dynamically adjusts font size

vertically centers text

prevents overflow outside bubble

This improves visual quality significantly.

Technologies Used
Python
PyTorch
Torchvision
OpenCV
PIL
MangaOCR
Ollama
Qwen2.5
Final Results

The system can:

Detect manga dialogue regions

Extract Japanese text

Translate dialogue into natural English

Insert translated text back into the manga image

Result:
Fully automated manga page translation pipeline.

Future Improvements

Possible upgrades:

Bubble segmentation

Instead of rectangular removal.

Text inpainting

Preserve bubble background.

Multi-bubble context translation

Translate entire panel dialogue.

Sound-effect recognition

Handle stylized SFX text.

Project Outcome

Successfully built a working manga translator prototype combining:

computer vision

OCR

large language models

automated typesetting

This demonstrates an end-to-end AI application integrating multiple ML components.
