from transformers import StoppingCriteria
import torch

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [835, 2799, 4080, 29901]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids




def generation_utils(query, args, tokenizer):
    """Format the queries for dialogue generation and set a stopping criterion. Edit this function to run other models."""

    if args.hf_chat_template:
        processed = [tokenizer.apply_chat_template([{"role": "user", "content": q.strip("\n")}], tokenize=False, add_generation_prompt=True) for q in query]
    else:
        # default formatting
        processed = [f"{tokenizer.bos_token}\nuser: {q}{tokenizer.eos_token}\nassistant:" for q in query]

        # custom formatting
        if "microsoft/phi" in args.model.lower():
            processed = [f"user: {q}\nassistant:" for q in query]


    # default stopping
    stop = [EosListStoppingCriteria(tokenizer.encode(tokenizer.eos_token))]

    # custom stopping
    if "wizardmath-7b-v1.0" in args.model.lower():
        stop = [EosListStoppingCriteria([13,   829, 29879, 29958])]
    elif "microsoft/phi" in args.model.lower():
        stop = [EosListStoppingCriteria(tokenizer.encode("\nuser:"))]

    return processed, stop