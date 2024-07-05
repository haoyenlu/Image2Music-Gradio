from PIL import Image
import requests
from transformers import AutoProcessor, MusicgenForConditionalGeneration, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch




class Pipeline:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.llava_model = LlavaForConditionalGeneration.from_pretrained("./llava-hf",local_files_only=True,load_in_8bit=True)
        self.llava_processor = AutoProcessor.from_pretrained("./llava-hf",local_files_only=True,load_in_8bit=True)
        self.musicgen_model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-small").to(self.device)
        self.musicgen_processor = AutoProcessor.from_pretrained("./musicgen-small")


    def musicgen(self,prompt, max_num_token=200):
        inputs = self.musicgen_processor(text=[prompt],padding=True,return_tensors="pt").to(self.device)
        result = self.musicgen_model.generate(**inputs, do_sample=True, guidance_scale=3,max_new_tokens=max_num_token)
        return {"audio":result[0].numpy(), "sample_rate":self.musicgen_model.config.audio_encoder.sampling_rate}


    def llava(self,image,prompt,max_num_token=20):
        full_prompt = f"USER: <image>\\n{prompt} ASSISTANT:"
        inputs = self.llava_processor(text=full_prompt, images=image, return_tensors="pt").to(self.device)
        generate_ids = self.llava_model.generate(**inputs, max_new_tokens=max_num_token)
        result = self.llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[1]
        return result


