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

class LlmModel:
    """
    Class for loading Large Language Models.

    Args:
        model_name (str): Name of the pre-trained model.
        model_type (str): Type of the model, either "seq2seq" or another.
        hf_access_token (str): Hugging Face token required to access some llm pretrains that need authentication
        quantize (bool, optional): Whether to quantize the model. Defaults to False.

    Attributes:
        model_name (str): Name of the pre-trained model.
        model_type (str): Type of the model.
        quantize (bool): Whether the model is quantized.
        model: Loaded model object.
    """
    
    def __init__(self, model_name: str, model_type: str,
                 peft_model_path: str = None,
                 hf_access_token: str = None,
                 quantize: bool = False,
                 training: bool = False):
        
        self.model_name = model_name
        self.model_type = model_type
        self.hf_access_token = hf_access_token
        self.peft_model_path = peft_model_path
        self.quantize = quantize
        self.training = training

    def get_model(self):
        """
        Load and return the pre-trained model.

        Returns:
            model: Loaded model object.
            tokenizer: Tokenizer model for input sequence.
        """
        try:
            bnb_config = self.get_quantization_config() if self.quantize else None
            
            if self.model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    token=self.hf_access_token,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    token=self.hf_access_token,
                    device_map="auto"
                )
                
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model: {e}")
            
            return None

    def get_quantization_config(self):
        """
        Return the quantization configuration.

        Returns:
            BitsAndBytesConfig: Quantization configuration.
        """
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
    def get_peft_model(self, r=16, lora_alpha=32,
                        target_modules="all-linear",
                        lora_dropout=0.05, bias="none"):
        """
        Return the model after apply peft (Parameter-Efficient Fine-tuning).
        
        - r, lora_alpham, target_modules, lora_dropout, bias: Configs for LoRA
        
        Because peft is only use when you need do finetune a model for specific needs so:
        + If finetuning is required: We will prepare model for k-bit training and applying peft.
        + Else: Just apply a pretrain peft to our model for immediate use.
        
        Returns:
            peft_model: Model after applying peft.
            tokenizer: Tokenizer model for input sequence.
        """
        model, tokenizer = self.get_model()
        if self.training == True:
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            lora_config = self.get_lora_config()
            peft_model = get_peft_model(model, lora_config)
        else:
            peft_model = PeftModel.from_pretrained(model, self.peft_model_path)
        
        return peft_model,tokenizer
        
    def get_lora_config(self, r=16, lora_alpha=32,
                        target_modules="all-linear",
                        lora_dropout=0.05, bias="none"):
        """
        Return the lora configuration.

        Returns:
            LoraConfig: LoRA configuration for applying PEFT.
        """
        
        task_type = "SEQ_2_SEQ_LM" if self.model_type == "seq2seq" else "CAUSAL_LM"
        lora_config = LoraConfig(
            r=16, # Rank
            lora_alpha=32,        #ΔW=α/r | ΔW represents the change in weights due to LoRA. α as a knob that controls how much influence the LoRA activations have on the overall model.
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type=task_type,
        )

        return lora_config
