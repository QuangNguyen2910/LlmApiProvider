from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import argparse
from transformers import (
    AutoConfig, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, PeftConfig, PeftModel, get_peft_model,
    prepare_model_for_kbit_training, TaskType
)
import torch
import torch.nn as nn
import json
import os
import bitsandbytes as bnb
import transformers
import pandas as pd
import numpy as np
import wandb
from Model.llm_model import LlmModel


with open('server_config.txt', 'r') as f:
    lines = f.readlines()
    MODEL_NAME = lines[0].strip()
    PEFT_MODEL_PATH = lines[1].strip()
    MODEL_TYPE = lines[2].strip()
    QUANTIZE = lines[3].strip()
    HF_ACCESS_TOKEN = lines[4].strip()
    TRAINING = lines[5].strip()

if PEFT_MODEL_PATH == 'None':
    PEFT_MODEL_PATH = None

if QUANTIZE == 'True':
    QUANTIZE = True
else:
    QUANTIZE = False

if TRAINING == 'True':
    TRAINING = True
else:
    TRAINING = False

if HF_ACCESS_TOKEN == 'None':
    HF_ACCESS_TOKEN = None
    
model = LlmModel(model_name=MODEL_NAME,
                 peft_model_path=PEFT_MODEL_PATH,
                 model_type=MODEL_TYPE,
                 quantize=QUANTIZE,
                 hf_access_token=HF_ACCESS_TOKEN)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, tokenizer = model.get_model()

generation_config = model.generation_config
generation_config.max_new_tokens = 80
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.do_sample = True

def run(question):

    if MODEL_TYPE == "seq2seq":
        prompt = f"""
        Answer the following question base on the contexts below.
    
        Question: {question}
    
        Answer: """
    else:
        prompt = f"""
        <|im_begin|>system
        Answer the following question base on the contexts below.<|im_end|>
        <|im_begin|>user
        Question:
        {question}
        Answer:<|im_end|>
        <|im_begin|>assitant"""
    
    encoding = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )
    if MODEL_TYPE == "seq2seq":
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    else:
        answer = tokenizer.batch_decode(outputs[:, encoding.input_ids.shape[1]:], skip_special_tokens=True)[0].rstrip("<|im_end|>")
    
    return answer



app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "it works!"}


@app.post("/llm_chat")
async def llm_chat(req: Request):
    jsonFromRequest = await req.json();

    message = jsonFromRequest["message"]

    res = {
        "answer": run(message)
    }

    return res

ngrok.set_auth_token("2fgwIieOo18tKdnOU1FHSkSewKL_3UYJjhCi1uy3c6tteEdTd")

ngrok_tunnel = ngrok.connect(5000)
print('Public URL:', f"{ngrok_tunnel.public_url}/llm_chat")
nest_asyncio.apply()
uvicorn.run(app, port=5000)
