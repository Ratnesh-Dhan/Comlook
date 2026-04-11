from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self):
        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #local_files_only=True 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda") #.to("cuda")

    # def detect_language(self, text):
    #     for ch in text:
    #         if '\u3040' <= ch <= '\u30ff':
    #             return "jpn_Jpan"
    #         if '\u4e00' <= ch <= '\u9fff':
    #             return "zho_Hans"
    #     return "jpn_Jpan"
    
    def detect_language(self, text):    
        for ch in text:
            code = ord(ch)

            # Hiragana / Katakana → Japanese
            if 0x3040 <= code <= 0x30FF:
                return "jpn_Jpan"

            # CJK only → Chinese fallback
            if 0x4E00 <= code <= 0x9FFF:
                return "zho_Hans"

        return "jpn_Jpan"

    def translate_nllb(self, text):
        self.tokenizer.src_lang = self.detect_language(text)
        inputs = self.tokenizer(text, return_tensors='pt').to("cuda") # to("cuda")
 
        translated_tokens =self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("eng_Latn"),
            # forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"],
            # max_length=256, #128
            max_length=80,
            num_beams=4,
        ) 
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0], text

        # Example usage
        # japanese_text = "こんにちは、世界！"
        # english = translate_nllb(japanese_text)
        # print(english)  