from tqdm import tqdm
import argparse
import os
import json
from utils.openai_utils import OpenAI
# from utils.togetherai_utils import TogetherBaseEngine
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from utils.generation_utils import generation_utils
from datasets import load_dataset, load_from_disk
import torch



def generate_answers(data, template, model, tokenizer=None):
    outputs = []
    for sample in tqdm(data):
        chapters = sample["chapter"]
        questions = sample["question"]
        sample["template"] = [template]*len(questions)
        query = [template.replace("{{QUESTION}}", q).replace("{{CHAPTER}}", c) for (q,c) in zip(questions, chapters)]

        if "openai/gpt" in args.model:
            assert args.batch_size == 1
            response = [model.complete(query)]
        elif args.togetherapi:
            assert args.batch_size == 1
            prompt="<s>user: "+ query[0] + "</s>\nassistant: "
            response = model.safe_completion(prompt, check_prompt=False)["content"]
        else:
            query, stop = generation_utils(query, args, tokenizer)
            inputs = tokenizer(query, add_special_tokens=False, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                out = model.generate(inputs=inputs["input_ids"], attention_mask = inputs["attention_mask"], pad_token_id=tokenizer.eos_token_id, stopping_criteria=stop, max_new_tokens=800)
            out = out[: , inputs["input_ids"].shape[1]:]
            response = tokenizer.batch_decode(out, skip_special_tokens=True)
        sample["output"] = response
        sample["model"] =[args.model]*len(questions)
        sample["closedbook_eval"] = [args.closedbook]*len(questions)
        sample["hf_chat_template"] = [args.hf_chat_template]*len(questions)
        sample["bnb4bit"] = [args.bnb4bit]*len(questions)
        outputs+= [ {k: sample[k][i] for k in sample.keys()} for i in range(len(sample["output"]))]
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="princeton-nlp/Llemma-7B-32K-MathMix", type=str, help="Generator model")
    parser.add_argument("--output_dir", default="tutoreval/generations", type=str, help="output simulations")
    parser.add_argument("--closedbook", action="store_true", help="output simulations")
    parser.add_argument("--hf_chat_template", action="store_true", help="If True, uses the chat template from tokenizer. If False, uses defaut user/assistant formatting and allows custom implementations")
    parser.add_argument("--togetherapi", action="store_true", help="use the TogetherAI API")
    parser.add_argument("--rope_theta", default=-1, type=int, help="Set a higher RoPE theta for context window extension. If set to -1, use the pre-trained config value.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size used during generation. Only for locally run models")
    parser.add_argument("--ddp_worldsize", default=1, type=int, help="For data parallel. Sets the number of parallel instances")
    parser.add_argument("--ddp_rank", default=0, type=int, help="For data parallel. Set this to the data fragment to use for generation. Value should be in range(args.ddp_worldsize)")
    parser.add_argument("--bnb4bit", action="store_true", help="Use 4 bit quantization")
    
    args = parser.parse_args()


    # load data
    try:
        data = load_dataset("princeton-nlp/TutorEval")["train"]
    except:
        try:    
            data = load_from_disk("tutoreval/tutoreval_dataset/train")
        except:
            print("Please download the dataset from princeton-nlp/TutorEval and save it under tutoreval/tutoreval_dataset")
            exit()

    if args.closedbook:
        data = data.filter(lambda x: x["closed_book"])
        with open("tutoreval/templates/closedbook_generation_template.txt", "r") as f:
            template = f.read()
    else:
        with open("tutoreval/templates/generation_template.txt", "r") as f:
            template = f.read()

    if args.ddp_worldsize > 1:
        assert args.ddp_rank in range(args.ddp_worldsize)
        data = data.select(list(range(args.ddp_rank, len(data), args.ddp_worldsize)))
    data = torch.utils.data.DataLoader(data, batch_size = args.batch_size, shuffle=False)


    if "openai/gpt" in args.model:                                              # openai api
        # examples: openai/gpt-3.5-turbo-16k openai/gpt-4-1106-preview
        engine=args.model.split("/")[1]
        print(engine)
        args.system_prompt = "You are a helpful science teacher interacting with a keen student. You try your utmost to answer the student's questions and to encourage the student to learn further. You are also very careful to provide clear, accurate, and factual answers, as you must not mislead the student in any way"
        model = OpenAI(model=engine, system_prompt=args.system_prompt)
        print(model.complete(["Hello! Introduce yourself please!"]))
        tokenizer = None
        args.batch_size = 1
    elif args.togetherapi:
        model = TogetherBaseEngine(args.model)
        print(model.complete(["user: Hello! Introduce yourself please!</s>\nassistant: "]))
        tokenizer = None
        args.batch_size = 1

    else:            
        config = AutoConfig.from_pretrained(args.model)
        config.max_new_tokens = 800
        config.dtype=torch.bfloat16
        config.do_sample = False
        config.use_cache=True
        if args.rope_theta != -1:
            config.rope_theta=args.rope_theta
            print(f"Setting RoPE theta = {args.rope_theta}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token

        if args.bnb4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
                )            

        model.eval()
    
    outputs = generate_answers(data, template, model, tokenizer)

    # postprocessing
    for out in outputs:
        for k in out.keys():
            out[k] = out[k].item() if type(out[k]) == torch.Tensor else out[k]
    
    # save
    if args.closedbook:
        base_save = f"{args.output_dir}/closedbook"
    else:
        base_save = f"{args.output_dir}/openbook"

    if args.ddp_worldsize > 1:
        save_dir = f"{base_save}/{args.model}_{args.ddp_rank}_of_{args.ddp_worldsize}.json"
    else:
        save_dir = f"{base_save}/{args.model}.json"

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    with open(save_dir, "w+") as f:
        f.write(json.dumps(outputs, indent=4))
