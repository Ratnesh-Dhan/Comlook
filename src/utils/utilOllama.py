import ollama

class Utilollama:
    def __init__(self, prompts):
        self.client = ollama.Client(host="http://127.0.0.1:11434")
        self.prompts = prompts
    
    def joined_text(self, texts):
        return  "\n".join([f"Bubble {i}: {text}" for i ,text in enumerate(texts)])
    
    def ollam_chat(self, joined_text):
        response = self.client.chat( model="huihui_ai/phi4-mini-abliterated", #"phi4-mini:latest",
                    messages=[
                        {"role": "system", "content": self.prompts.system_prompt},
                        {"role": "user", "content": joined_text}
                    ]
                )
        return response['message']['content'].strip().split("\n")
    
    def ollama_generate(self, texts):
        # system_prompt = self.prompts.get_system_prompt(self.joined_text(texts))
        system_prompt = self.prompts.get_improved_prompt(self.joined_text(texts))
        print("THIS IS JOINT TEXT : ",self.joined_text(texts))
        response  = self.client.generate(
            # model='huihui_ai/phi4-mini-abliterated',
            model='richardyoung/qwen2.5-7b-instruct-abliterated',
            prompt= system_prompt,
            options={
                "temperature":0.0,
                "top_p": 0.9,
                "num_predict": 256,
                "repeat_penalty": 1.2,
                "stop": [f"Bubble {len(texts)}:"],
                }
        )
        # print("this is response : ",response["response"])
        return response['response'].strip().split("\n")