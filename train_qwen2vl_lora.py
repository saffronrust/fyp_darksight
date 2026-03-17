import json, os
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

BASE = "Qwen/Qwen2-VL-2B-Instruct"

def read_split(path): return json.load(open(path, "r", encoding="utf-8"))

@dataclass
class ChatCollator:
    processor: Any
    prompt: str

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Build per-sample messages
        imgs = [Image.open(b["image"]).convert("RGB") for b in batch]
        msgs_full, msgs_prompt = [], []
        for b in batch:
            user = {"role":"user","content":[{"type":"image","image":Image.open(b["image"]).convert("RGB")},
                                             {"type":"text","text":self.prompt}]}
            assistant = {"role":"assistant","content":b["caption"]}
            msgs_full.append([user, assistant])   # full conversation
            msgs_prompt.append([user])            # prompt only

        # Render to chat text (with the same template both times)
        texts_full   = [self.processor.apply_chat_template(m, add_generation_prompt=False, tokenize=False) for m in msgs_full]
        texts_prompt = [self.processor.apply_chat_template(m, add_generation_prompt=True,  tokenize=False) for m in msgs_prompt]

        # Tokenize the full sequences (includes image placeholders)
        enc = self.processor(text=texts_full, images=imgs, padding=True, return_tensors="pt")
        # Tokenize prompt-only to know how much to mask with -100
        tok = self.processor.tokenizer
        enc_prompt = tok(texts_prompt, padding=True, return_tensors="pt")

        labels = enc["input_ids"].clone()
        # mask out everything up to the end of the prompt for each sample
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()
        for i, L in enumerate(prompt_lens):
            labels[i, :L] = -100
        labels[labels == tok.pad_token_id] = -100
        enc["labels"] = labels
        return enc

def print_trainable(model):
    t, n = 0, 0
    for p in model.parameters():
        n += p.numel()
        if p.requires_grad: t += p.numel()
    print(f"Trainable: {t/1e6:.2f}M / {n/1e6:.2f}M ({100*t/n:.2f}%)")

def main(
    train_json="splits/train.json",
    val_json="splits/val.json",
    out_dir="qwen2vl-dog-lora",
    lr=1e-4, epochs=3, bsz=2, grad_acc=8, lora_r=16, lora_alpha=16, lora_drop=0.05
):
    print(torch.cuda.is_available())
    ds = DatasetDict({
        "train": Dataset.from_list(read_split(train_json)),
        "validation": Dataset.from_list(read_split(val_json)),
    })

    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

    # 4-bit QLoRA
    qconf = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    	BASE, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False  # needed for gradient checkpointing in Trainer
    model.gradient_checkpointing_enable()

    # freeze vision + projector (tiny data → avoid overfitting)
    for name, p in model.named_parameters():
        if any(k in name for k in ["visual", "vision", "multi_modal_projector"]):
            p.requires_grad = False

    # LoRA on language blocks
    lconf = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_drop, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lconf)
    print_trainable(model)

    collator = ChatCollator(processor=proc, prompt="Describe this image in detail.")

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        gradient_accumulation_steps=grad_acc,
        warmup_ratio=0.1,
        weight_decay=0.05,
        logging_steps=20,
        # evaluation_strategy="steps",
        eval_steps=100,
        # save_strategy="steps",
        save_steps=100,
        # save_total_limit=2,
        # load_best_model_at_end=True,
        # metric_for_best_model="loss",
        # greater_is_better=False,
        fp16=True,
        # report_to="none",
        remove_unused_columns=False,  # keep pixel_values
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(out_dir)
    proc.save_pretrained(out_dir)
    print("Done. Adapter saved to", out_dir)

if __name__ == "__main__":
    main()