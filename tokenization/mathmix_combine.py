from datasets import Dataset, load_from_disk, concatenate_datasets
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--tutorchat", default="data/tokenized_tutorchat_stem_llama", type=str)
parser.add_argument("--metamath", default="data/tokenized_metamath_concat10_llama", type=str)
parser.add_argument("--save_dir", default="data/mathmix_llama")
args = parser.parse_args()

tutorchat = load_from_disk(args.tutorchat)["train"]
metamath = load_from_disk(args.metamath)

to_remove = [k for k in tutorchat.features.keys() if k not in ["input_ids", "attention_mask", "labels", "processed_conversation"]]
tutorchat = tutorchat.remove_columns(to_remove)
tutorchat.rename_column("processed_conversation", "text")

to_remove = [k for k in metamath.features.keys() if k not in ["input_ids", "attention_mask", "labels", "text"]]
metamath = metamath.remove_columns(to_remove)

mathmix = concatenate_datasets([tutorchat, metamath])
mathmix = mathmix.shuffle(seed=42)
mathmix.save_to_disk(args.save_dir)