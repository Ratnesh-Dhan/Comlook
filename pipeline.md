manga page
↓
detect text boxes
↓
OCR
↓
LLM translate
↓
erase original text
↓
draw translated text
↓
translated manga page

# We are going to change the entire pipeline because even after making it optimized the system is censoring some texts because of expicit contents. So we are going with this now :-

looping manga pages
↓
detect text boxes
↓
OCR
↓
Translate using translation model (NLLB from facebook)
↓
then rephraseing the text using small LLM to sound more natural or manga style(Google FLAN-T5 Base / Small)
↓
erase original text
↓
draw translated text
↓
translated manga page


# Piplne for dummies
detect text boxes
↓
MangaOCR
↓
NLLB
↓
raw English
↓
FLAN-T5
↓
natural dialogue
