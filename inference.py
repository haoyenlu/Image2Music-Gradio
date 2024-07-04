from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from PIL import Image
import requests
from transformers import AutoProcessor, MusicgenForConditionalGeneration, LlavaForConditionalGeneration
from functools import cache
import os
import threading


app = FastAPI()


class MusicGenRequestItem(BaseModel):
    prompt: str
    max_num_token: int | None = None

class LlavaRequestItem(BaseModel):
    url: str
    prompt: str
    max_num_token: int | None = None

@cache
def load_llava_model():
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    return model , processor



@app.get('/health')
def health_check():
    return {"message":"Connect successfully."}


@app.post('/musicgen')
def musicgen(item: MusicGenRequestItem):
    return item


@app.post('/llava')
def llava(item: LlavaRequestItem):
    image = Image.open(requests.get(item.url,stream=True).raw)
    full_prompt = f"USER: <image>\\n{item.prompt} ASSISTANT:"
    model, processor = load_llava_model()
    inputs = processor(text=full_prompt, images=image, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=item.max_num_token)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[1]
    return response




if __name__ == '__main__':
    config = uvicorn.Config(app=app,host="0.0.0.0",port=int(os.environ['MODEL_PORT']))
    server = uvicorn.Server(config=config)
    thread = threading.Thread(target=server.run)
    thread.start()
