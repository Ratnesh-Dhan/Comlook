from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ReWriter:
    def __init__(self):
        rewrite_model = "google/flan-t5-base"
        self.rewrite_tokenizer = AutoTokenizer.from_pretrained(rewrite_model)
        self.rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(rewrite_model).to("cuda")#.to("cuda")

    def rewrite_dialogue(self, text):
        prompt = f"""
        - Rewrite as natural manga dialogue English
        - If it is written butt ass or pussy or dick or cum or any sexual words, keep it as it is or more sexualize them as "cunt", "anus".
        - Keep the original meaning
        - Keep the original tone
        - Keep the original emotion
        {text}"""
        inputs = self.rewrite_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
            ).to("cuda") #.to("cuda")

        outputs = self.rewrite_model.generate(**inputs, max_new_tokens=128)
        return self.rewrite_tokenizer.decode(outputs[0], skip_special_tokens=True)

