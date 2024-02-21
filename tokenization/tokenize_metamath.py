from tqdm import tqdm
import random
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
import json
import argparse



def concat_conversations(dataset, num_concat, tokenizer):
    l = len(dataset["query"])

    new_dataset = {k: [] for k in dataset.keys()}
    new_dataset["text"] = []
    new_dataset["input_ids"] = []
    new_dataset["attention_mask"] = []
    new_dataset["labels"] = []

    for k in tqdm(range(0, l, num_concat), desc="Concatenating and tokenizing"):
        for key in dataset.keys():
            new_dataset[key].append([dataset[key][i] for i in range(k, k+num_concat)])

        options = [
            ("\nquestion: ", "\nanswer: "), 
            ("\nQuestion: ", "\nAnswer: "), 
            ("\nProblem: ", "\nSolution: "), 
            ("\nproblem: ", "\nsolution: "), 
            ("\nuser: ", "\nassistant: "), 
            ("\nassistant: ", "\nuser: ")
            ]

        turn0, turn1 = rng.sample(options, 1)[0]
        conversation = tokenizer.bos_token
        for i in range(k, k + num_concat):
            conversation+= "".join([turn0, dataset["query"][i], f"{tokenizer.eos_token}", turn1, dataset["response"][i], f"{tokenizer.eos_token}"])
        new_dataset["text"].append(conversation)
        new_dataset["input_ids"].append(tokenizer.encode(conversation, add_special_tokens=False))
        new_dataset["attention_mask"].append([1]*len(new_dataset["input_ids"][-1]))
        new_dataset["labels"].append(new_dataset["input_ids"][-1])
    return new_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf", help="Choose the HF tokenizer")
    parser.add_argument("--num_concat", type=int, default=10, help="Number of MetaMath samples to concatenate")
    parser.add_argument("--save_dir", type=str, default="data/metamath_concat10_llama", help="Directory for saving the HF dataset")
    args = parser.parse_args()
    rng = random.Random(4)


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    data = load_dataset("meta-math/MetaMathQA")
    data = data.shuffle(seed=42)
    data = data["train"].to_dict()

    tokenized = concat_conversations(data, args.num_concat, tokenizer)
    tokenized = Dataset.from_dict(tokenized)
    tokenized.save_to_disk(args.save_dir)
