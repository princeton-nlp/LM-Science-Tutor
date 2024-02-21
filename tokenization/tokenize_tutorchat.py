import os
import json
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
import random
import argparse



def clean_and_assign(name, all_text):
    """takes a simulated conversation, applies basic cleaning, and assigns student/teacher roles to help split the conversation into a dialogue"""
    # truncate between first and last occurrence of ### 
    first = all_text.find("###")
    if first <= len(all_text)//2:
        all_text = all_text[first+3:]
    last = all_text.rfind("###")
    if last >= len(all_text)//2:
        all_text = all_text[:last]
    all_text = all_text.replace("###", "").strip("\n ")

    # assign roles 
    if "generateexam" in name:
        key0, key1 = "QUESTION", "ANSWER"
        options = [
            ("\nquestion: ", "\nanswer: "), 
            ("\nuser: ", "\nassistant: "), 
            ("\nassistant: ", "\nuser: ")
            ]
        turn0, turn1 = options[rng.sample([0,1,2],1)[0]]
    elif "studentstart" in name:
        key0, key1 = "STUDENT", "TEACHER"
        turn0, turn1 = "\nuser: ", "\nassistant: "
    elif "teacherstart" in name:
        key0, key1 = "TEACHER", "STUDENT"
        turn0, turn1 = "\nassistant: ", "\nuser: "

    # ignore badly formatted texts
    if key0 not in all_text:
        return 
    return key0, key1, turn0, turn1, all_text
    

def tokenize(dialogue, tokenizer, args):
    input_ids = []
    labels = []
    processed_conversation = ""
    if dialogue["mode"] == "openbook":
        for m, turn in enumerate(dialogue["conversation"]):
            if m == 0:
                turn_text = turn + f"{tokenizer.eos_token}\n{tokenizer.bos_token}"
                tokenized_turn = tokenizer.encode(turn_text, add_special_tokens=False)
                labels += [-100]*len(tokenized_turn)
            elif m % 2 == 0:
                turn_text = "\nassistant: " + turn + f"{tokenizer.eos_token}"
                tokenized_turn = tokenizer.encode(turn_text, add_special_tokens=False)
                labels += tokenized_turn
            elif m % 2 == 1:
                turn_text = "\nuser: " + turn + f"{tokenizer.eos_token}"
                tokenized_turn = tokenizer.encode(turn_text, add_special_tokens=False)
                labels += [-100]*len(tokenized_turn) 
            input_ids += tokenized_turn
            processed_conversation += turn_text

    elif dialogue["mode"] == "closedbook":
        for m, turn in enumerate(dialogue["conversation"]):
            if m % 2 == 0:
                turn_text = "\nassistant: " + turn +  f"{tokenizer.eos_token}"
                if m == 0:
                    turn_text = f"{tokenizer.bos_token}"+turn_text
                tokenized_turn = tokenizer.encode(turn_text, add_special_tokens=False)
                labels += tokenized_turn
            elif m % 2 == 1:
                turn_text = "\nuser: " + turn + f"{tokenizer.eos_token}"
                tokenized_turn = tokenizer.encode(turn_text, add_special_tokens=False)
                labels += [-100]*len(tokenized_turn)
            input_ids += tokenized_turn
            processed_conversation += turn_text

    elif dialogue["mode"] == "singleturn":
        name = dialogue["name"]
        # get chapter text and make labels
        if "studentstart" in name:
            turn_text = dialogue["conversation"][0] + f"{tokenizer.eos_token}\n{tokenizer.bos_token}"
            tokenized_turn = tokenizer.encode(turn_text, add_special_tokens=False)
            input_ids += tokenized_turn
            labels += [-100]*len(tokenized_turn)
            processed_conversation += turn_text
        else:
            processed_conversation += f"{tokenizer.bos_token}"
        
        all_text = dialogue["conversation"][-1]
        key0, key1, turn0, turn1, all_text = clean_and_assign(name, all_text)

        # split by keys
        qa_pairs = all_text.split(key0)
        qa_lists = [s.split(key1) for s in qa_pairs]
        qa_lists = [s for s in qa_lists if len(s) == 2]
        qa_flat = [t.strip(": \n") for s in qa_lists for t in s]
        qa_flat = [t for t in qa_flat if t != ""]

        # add turns and make roles
        for m, turn in enumerate(qa_flat):
            if m % 2 == 0:
                turn_text = turn0 + turn + f"{tokenizer.eos_token}"
            else:
                turn_text = turn1 + turn + f"{tokenizer.eos_token}"
            processed_conversation+= turn_text
            tokenized_turn = tokenizer.encode(turn_text, add_special_tokens=False)
            input_ids += tokenized_turn
            if "studentstart" in name and m % 2 == 0:
                labels += [-100]*len(tokenized_turn)
            else:
                labels += tokenized_turn

    dialogue["input_ids"] = input_ids
    dialogue["attention_mask"] = [1]*len(input_ids)
    dialogue["labels"] = labels
    dialogue["processed_conversation"] = processed_conversation
    return dialogue




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf", help="Choose the HF tokenizer")
    parser.add_argument("--stem_only", action="store_true", help="Tokenize only STEM domains")
    parser.add_argument("--save_dir", type=str, default="data/tokenized_tutorchat_llama", help="Directory for saving the HF dataset")
    args = parser.parse_args()

    if args.stem_only:
        domains = ["bio", "chem", "eng", "geo", "math", "med", "phys", "stats"]
    else:
        domains = []

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(4)


    all_dialogues = load_dataset("princeton-nlp/TutorChat")
    dialogues = all_dialogues.filter(lambda x : x["textbook_folder"].split("/")[1] in domains, num_proc=8) if domains != [] else all_dialogues
    validation = dialogues["validation"]
    validation = validation.map(lambda x: tokenize(x, tokenizer, args), num_proc=4)
    train = dialogues["train"]
    train = train.map(lambda x: tokenize(x, tokenizer, args), num_proc=4)


    tokenized = DatasetDict({
        "train": train, 
        "validation": validation
    })
    
    tokenized.save_to_disk(args.save_dir)
