import argparse 
import json
import os
import pandas as pd 
from rich import print

def get_all_results(df):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="tutoreval/generations")
    parser.add_argument("--results_dir", default="tutoreval/results")
    parser.add_argument("--closedbook", action="store_true")
    parser.add_argument("--model", default="princeton-nlp/Llemma-7B-32K-MathMix")
    args = parser.parse_args()

    if args.closedbook:
        file = f"{args.output_dir}/closedbook/{args.model}.json"
        results_file = f"{args.results_dir}/closedbook/{args.model}.json"
    else:
        file = f"{args.output_dir}/openbook/{args.model}.json"
        results_file = f"{args.results_dir}/openbook/{args.model}.json"

    with open(file) as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    
    #scale 
    df["presentation"] = 100*df["presentation"]/3
    df["correctness"] = 100*df["correctness"]/3


    results = {
        "total": df["correctness"].mean(),
        "presentation_score": df["presentation"].mean(),
    }
    # scientific domain 
    results = results | df.groupby(["domain"])["correctness"].mean().to_dict()

    # difficulty
    results = results | df.groupby(["difficulty"])["correctness"].mean().to_dict()

    # misleading
    results["misleading_questions"] = df[df["misleading_question"]]["correctness"].mean()

    # answer_in_chapter
    results["answer_in_chapter"] = df[df["answer_in_chapter"]]["correctness"].mean()


    print(results)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
