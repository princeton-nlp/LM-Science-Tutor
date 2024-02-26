import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="princeton-nlp/Llemma-7B-32K-MathMix", type=str, help="Generator model")
    parser.add_argument("--dir", default="tutoreval/generations", type=str, help="output simulations")
    parser.add_argument("--ddp_worldsize", default=1, type=int, help="For data parallel. Sets the number of parallel instances")
    parser.add_argument("--closedbook", action="store_true", help="output simulations")

    args = parser.parse_args()
    
    if args.ddp_worldsize == 1:
        print("Generations merged.")
        exit()

    if args.closedbook:
        files = [f"{args.dir}/closedbook/{args.model}_{rank}_of_{args.ddp_worldsize}.json" for rank in range(args.ddp_worldsize)]
        save_file = f"{args.dir}/closedbook/{args.model}.json"
    else:
        files = [f"{args.dir}/openbook/{args.model}_{rank}_of_{args.ddp_worldsize}.json" for rank in range(args.ddp_worldsize)]
        save_file = f"{args.dir}/openbook/{args.model}.json"

    all_generations = []
    for file in files:
        with open(file) as f:
            all_generations += json.load(f)
    


    with open(save_file, "w") as f:
        json.dump(all_generations, f, indent=4)
    print("Generations merged.")