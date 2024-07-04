from pydantic import BaseModel
from PIL import Image
import requests
from transformers import AutoProcessor, MusicgenForConditionalGeneration, LlavaForConditionalGeneration, BitsAndBytesConfig



llava_model = LlavaForConditionalGeneration.from_pretrained("./llava-hf",local_files_only=True,device_map="auto")
llava_processor = AutoProcessor.from_pretrained("./llava-hf",local_files_only=True,device_map="auto")
musicgen_model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-small")
musicgen_processor = AutoProcessor.from_pretrained("./musicgen-small")

class MusicGenRequestItem(BaseModel):
    prompt: str
    max_num_token: int 

class LlavaRequestItem(BaseModel):
    url: str
    prompt: str
    max_num_token: int 




def musicgen(item: MusicGenRequestItem):
    inputs = musicgen_processor(text=[item.prompt],padding=True,return_tensors="pt",)
    result = musicgen_model.generate(**inputs, do_sample=True, guidance_scale=3,max_new_tokens=item.max_num_token)
    return {"audio":result[0].numpy(), "sample_rate":musicgen_model.config.audio_encoder.sampling_rate}


def llava(item: LlavaRequestItem):
    image = Image.open(requests.get(item.url,stream=True).raw)
    full_prompt = f"USER: <image>\\n{item.prompt} ASSISTANT:"
    inputs = llava_processor(text=full_prompt, images=image, return_tensors="pt")
    generate_ids = llava_model.generate(**inputs, max_new_tokens=item.max_num_token)
    result = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[1]
    return result




# if __name__ == '__main__':
#     config = uvicorn.Config(app=app,host="0.0.0.0",port=int(os.environ['MODEL_PORT']))
#     server = uvicorn.Server(config=config)
#     thread = threading.Thread(target=server.run)
#     thread.start()
