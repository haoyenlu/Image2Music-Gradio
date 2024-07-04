from PIL import Image
import requests
from transformers import AutoProcessor, MusicgenForConditionalGeneration, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

llava_model = LlavaForConditionalGeneration.from_pretrained("./llava-hf",local_files_only=True,load_in_8bit=True).to(device)
llava_processor = AutoProcessor.from_pretrained("./llava-hf",local_files_only=True,load_in_8bit=True)
musicgen_model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-small").to(device)
musicgen_processor = AutoProcessor.from_pretrained("./musicgen-small")




def musicgen(prompt, max_num_token=200):
    inputs = musicgen_processor(text=[prompt],padding=True,return_tensors="pt").to(device)
    result = musicgen_model.generate(**inputs, do_sample=True, guidance_scale=3,max_new_tokens=max_num_token)
    return {"audio":result[0].numpy(), "sample_rate":musicgen_model.config.audio_encoder.sampling_rate}


def llava(url,prompt,max_num_token=20):
    image = Image.open(requests.get(url,stream=True).raw)
    full_prompt = f"USER: <image>\\n{prompt} ASSISTANT:"
    inputs = llava_processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    generate_ids = llava_model.generate(**inputs, max_new_tokens=max_num_token)
    result = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[1]
    return result


