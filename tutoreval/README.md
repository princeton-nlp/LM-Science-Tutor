## 🧑‍💻 Evaluating with TutorEval [work in progress]

### Requirements

Please install the following packages:

```python
pip install torch flash_attn transformers accelerate bitsandbytes datasets pandas openai rich
```

### ✍️ Generating LM tutor outputs

#### Basic usage 
`generate.py` constructs the LM tutor outputs for each question and saves them under `./openbook`, or `./closedbook` for TutorEval-ClosedBook. Use the HuggingFace model name or the path where the model is stored with the `--model` flag.

For example, to evaluate Llemma-7B-32K-MathMix on TutorEval:
```python
python -m tutoreval.generate --model princeton-nlp/Llemma-7B-32K-MathMix
```

Use the `--closedbook` flag for TutorEval-ClosedBook:
```python
python -m tutoreval.generate --model princeton-nlp/Llemma-7B-32K-MathMix --closedbook
```

#### Chat templates
By default, TutorEval formats the LM tutor's prompt as a `user/assistant` dialogue. Some HuggingFace models recommend using other chat templates. To use the HuggingFace chat templates, use the `--hf_chat_template` flag. For example, to evaluate Mistral-7B-Instruct-v0.2:
```python
python -m tutoreval.generate --model mistralai/Mistral-7B-Instruct-v0.2 --hf_chat_template
```

To set use custom dialogue formatting, we recommend editing `./utils/generation_utils.py`.

#### Model sharding and data parallel

To run larger models (e.g. [princeton-nlp/Llemma-34B-MathMix](https://huggingface.co/princeton-nlp/Llemma-34B-MathMix)), `generate.py` uses model parallel with `device_map="auto"`, so no modifications are required. 

Evaluating a 7B model on TutorEval takes approximately 4 hours on a single A100 GPU, so we also provide a basic data-parallel implementation. The number of data parallel instances is specified with the `--ddp_worldsize` flag, and the specific instance to be run is specified with `--ddp_rank`. 

`generate.sh` provides an easy interface for running several instances of `generate.py` on multiple GPUs. For example, if 4 GPUs are available, to evaluate Mistral-7B-v0.2 on TutorEval-ClosedBook, you can use 
```bash
MOD=mistralai/Mistral-7B-Instruct-v0.2 CLOSEDBOOK=true DDP=4 CHATTEMPLATE=true bash tutoreval/generate.sh
```

Note that `generate.sh` does not implement model parallel and data parallel simultaneously. Either the model will be sharded across all GPUs, or each GPU runs a separate instance of `generate.py`. If you have lots of GPUs available and you wish to use both methods at the same time, you can modify `generate.sh` to fit your needs by editing `CUDA_VISIBLE_DEVICES`.

### ☑️ Grading outputs with GPT-4
`grade.py` grades the LM tutor outputs and updates `./openbook` and `./closedbook` with the GPT-4 grades.
The script `./tutoreval/grade.sh` also provides some utilities for grading.
