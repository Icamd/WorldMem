"""Optional in-process model loading. Importing WorldMem does not import torch."""

import time


class HuggingFaceBackend:
    def __init__(self, model, *, device="auto", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        cpu = device == "cpu" or (device == "auto" and (not torch.cuda.is_available()))
        resolved_dtype = torch.float32 if cpu else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="cpu" if cpu else device,
            torch_dtype=resolved_dtype,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, messages, *, temperature, max_tokens, enable_thinking, top_p, top_k, min_p):
        from .clients import LLMResponse

        start = time.monotonic()
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        encoded = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_tokens = encoded.input_ids.shape[1]
        options = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            options["temperature"] = temperature
            options.update(
                {
                    k: v
                    for k, v in dict(top_p=top_p, top_k=top_k, min_p=min_p).items()
                    if v is not None
                }
            )
        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, **options)[0][prompt_tokens:]
        full = self.tokenizer.decode(generated, skip_special_tokens=True)
        content = full.rsplit("</think>", 1)[-1].strip()
        reasoning_tokens = 0
        if "</think>" in full:
            reasoning_tokens = len(self.tokenizer.encode(full.rsplit("</think>", 1)[0]))
        count = len(generated)
        usage = dict(
            prompt_tokens_est=prompt_tokens,
            completion_tokens_est=count,
            total_tokens_est=prompt_tokens + count,
            thinking_tokens_est=reasoning_tokens,
            visible_completion_tokens_est=max(0, count - reasoning_tokens),
        )
        if not content:
            raise RuntimeError("Local model produced an empty answer; increase max_tokens")
        return LLMResponse(
            content,
            usage,
            self.model.config.name_or_path,
            "length" if count >= max_tokens else "stop",
            time.monotonic() - start,
        )
