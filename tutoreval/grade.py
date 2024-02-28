from tqdm import tqdm
import argparse
import json
import re
from utils.openai_utils import OpenAI



def grade(grader_model, generations, args):
    for sample in tqdm(generations):
        prompt = args.template.format(**sample)
        grading_prompt=[prompt]
        try:
            sample['grading_out'] = grader_model.complete(grading_prompt)
            grades = [float(d) for d in re.findall(pattern=r':\s?(\d.*)/3', string=sample["grading_out"])]
            sample["presentation"] = grades[0]
            sample["correctness"] = grades[1]
            
        except:
            sample['grading_out'] = "ERROR"
            sample["presentation"] = 0
            sample["correctness"] = 0
    return generations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="princeton-nlp/Llemma-7B-32K-MathMix", help="Model whose outputs are evaluated")
    parser.add_argument("--dir", default="tutoreval/generations", help="Main directory where model outputs are stored")
    parser.add_argument("--closedbook", action="store_true", help="Selects the closedbook folder in main directory")
    parser.add_argument("--grader", default="gpt-4-1106-preview", help="OpenAI model used for grading")
    parser.add_argument("--ddp_worldsize", default=1, type=int, help="For data parallel. Sets the number of parallel instances")
    parser.add_argument("--ddp_rank", default=0, type=int, help="For data parallel. Set this to the data fragment to use for generation. Value should be in range(args.ddp_worldsize)")
    args = parser.parse_args()


    if args.closedbook:
        with open("tutoreval/templates/closedbook_grading_template.txt") as f:
            args.template = f.read()
        if args.ddp_worldsize > 1:
            generations_file = f"{args.dir}/closedbook/{args.model}_{args.ddp_rank}_of_{args.ddp_worldsize}.json"
        else:
            generations_file = f"{args.dir}/closedbook/{args.model}.json"
    else:
        with open("tutoreval/templates/grading_template.txt") as f:
            args.template = f.read()
        if args.ddp_worldsize > 1:
            generations_file = f"{args.dir}/openbook/{args.model}_{args.ddp_rank}_of_{args.ddp_worldsize}.json"
        else:
            generations_file = f"{args.dir}/openbook/{args.model}.json"
    with open(generations_file) as f:
        generations = json.load(f)

    grader_model = OpenAI(model=args.grader)
    print(grader_model.complete(["Hello! Introduce yourself please!"]))

    print("Grading")
    graded = grade(grader_model, generations, args)

    with open(generations_file, 'w') as file:
        json.dump(graded, file, indent=4)