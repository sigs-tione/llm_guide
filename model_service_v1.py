import os
from typing import Dict, List
import logging
import torch
import time

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, GenerationConfig

from mosec import Server, Worker, get_logger

# 模型路径
model_path = "Baichuan2-7B-Base"

logger = get_logger()
logger.setLevel(logging.DEBUG)

class Infer(Worker):
    def __init__(self) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model = model

    # 支持自动批处理
    def forward(self, data: List[Dict]) -> List[Dict]:
        start_time = time.time()
        batch_size = len(data)
        querys = [d["prompt"].strip() for d in data]
        inputs = self.tokenizer(querys, return_tensors="pt").to("cuda:0")
        start_generate_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            repetition_penalty=1.1
        )
            # inputs.input_ids,
            # attention_mask=inputs.attention_mask,
            # **self.generation_config,
        end_generate_time = time.time()
        results = []
        for i in range(batch_size):
            pred_full = self.tokenizer.decode(outputs.cpu()[i], skip_special_tokens=True)
            pred = pred_full[len(querys[i]):]
            results.append({"generated": pred.strip()})
        end_time = time.time()
        logger.info(
            f"time cost: total={1000*(end_time-start_time)}ms generate={1000*(end_generate_time-start_generate_time)}ms"
        )
        return results

if __name__ == '__main__':
    server = Server()
    server.append_worker(
        Infer, 
        num=1,
        max_batch_size=4,
        max_wait_time=100,
        )
    server.run()
