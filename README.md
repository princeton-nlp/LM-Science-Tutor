# LM-Science-Tutor

This is the official repository for *Language Models as Science Tutors*. 


## TutorEval
Download the TutorEval data from HuggingFace at [princeton-nlp/TutorEval](https://huggingface.co/datasets/princeton-nlp/TutorEval).

Coming soon:
- Evaluation script
- Human and GPT-4 gradings of the models evaluated in the paper



## TutorChat
Download the TutorChat data from HuggingFace at [princeton-nlp/TutorChat](https://huggingface.co/datasets/princeton-nlp/TutorChat).

### Textbook chapters 
Download the processed textbook chapters from HuggingFace at [princeton-nlp/TextbookChapter](https://huggingface.co/datasets/princeton-nlp/TextbookChapters). This dataset was obtained by scraping [libretexts.org](https://libretexts.org) and processing the cleaned HTML files with the HTML-to-LaTeX parser from [Openwebmath](https://github.com/keirp/OpenWebMath). 

### TutorChat processing
`./tokenization/tokenize_tutorchat.py` tokenizes TutorChat and creates training labels according to the recipe used to train `Llemma-7B-32K-MathMix`. Use the flag `--stem_only` to tokenize only the STEM split of TutorChat.

### MathMix
MathMix is a fine-tuning dataset composed of the STEM split of TutorChat and a processed version of [MetaMath](https://huggingface.co/datasets/meta-math/MetaMathQA). In `./tokenization`, we provide some scripts to re-create and tokenize MathMix.

`./tokenization/tokenize_metamath.py` tokenizes MetaMath by randomly concatenating question/answer pairs to form longer samples. Use the flag `--num_concat` to set the number of samples to concatenate. MathMix concatenates 10 samples at a time. 

`./mathmix_combine.py` concatenates and shuffles the tokenized TutorChat and MetaMath datasets created by the TutorChat and MetaMath tokenization scripts. Use the flags `--tutorchat` and `--metamath` to set the paths to your tokenized datasets.

## Models
Download our models from HuggingFace at [princeton-nlp/Llemma-7B-32K-MathMix](https://huggingface.co/princeton-nlp/Llemma-7B-32K-MathMix) and [princeton-nlp/Llemma-34B-MathMix](https://huggingface.co/princeton-nlp/Llemma-34B-MathMix).
