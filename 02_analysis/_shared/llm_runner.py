import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import codebook as cb

MODELS = {
    "qwen":      {"id": "Qwen/Qwen2.5-7B-Instruct",             "lab": "Alibaba",    "bs": 10},
    "ministral": {"id": "mistralai/Ministral-8B-Instruct-2410", "lab": "Mistral AI", "bs": 10},
    "phi4":      {"id": "microsoft/phi-4",                      "lab": "Microsoft",  "bs": 5},
}
FLUSH_EVERY = 20

def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="cuda:0", dtype=torch.bfloat16)
    model.eval(); return tok, model

def supports_system(tok):
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}], add_generation_prompt=True, tokenize=False)
        return True
    except Exception:
        return False

@torch.inference_mode()
def run_model(key, P, dest, bs):
    print(f"  loading {MODELS[key]['id']} ({MODELS[key]['lab']}) ...", flush=True)
    tok, model = load_model(MODELS[key]["id"]); sys_ok = supports_system(tok)
    print(f"    {torch.cuda.memory_allocated()/1e9:.1f} GB VRAM | batch {bs}", flush=True)
    P = P.assign(_len=P.sentence.str.len() + P.context_before.fillna("").str.len() + P.context_after.fillna("").str.len()).sort_values("_len", ascending=False)
    buf, t0, n = [], time.time(), 0
    def flush():
        nonlocal buf
        if buf: pd.DataFrame(buf).to_csv(dest, mode="a", header=not dest.exists(), index=False); buf = []
    starts, st = [], 0
    while st < len(P):
        nb = max(1, bs // 2) if P._len.iloc[st] > 1600 else bs
        starts.append((st, nb)); st += nb
    for b, (st, nb) in enumerate(starts, 1):
        ch = P.iloc[st:st + nb]; texts = []
        for _, r in ch.iterrows():
            user = cb.build_prompt(r)
            msgs = ([{"role": "system", "content": cb.SYSTEM}] if sys_ok else []) + [{"role": "user", "content": user if sys_ok else cb.SYSTEM + "\n\n" + user}]
            texts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        for (_, r), o in zip(ch.iterrows(), gen):
            reply = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            buf.append({"passage_id": r.passage_id, "model": key, **cb.parse_reply(reply), "raw_reply": " / ".join(reply.strip().splitlines())[:140]})
        n += len(ch)
        if b % FLUSH_EVERY == 0:
            flush(); rate = n / (time.time() - t0)
            print(f"    {n:,}/{len(P):,}  {rate:.2f}/s  eta {(len(P)-n)/rate/3600:.1f} h", flush=True)
    flush(); del model; torch.cuda.empty_cache()
