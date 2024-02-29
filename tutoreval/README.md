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
By default, TutorEval formats the LM tutor's prompt as a `user/assistant` dialogue. Some HuggingFace tokenizers have in-built chat templates that are recommended for the associated model. To use the HuggingFace chat templates. use the `--hf_chat_template` flag. For example, to evaluate Mistral-7B-Instruct-v0.2:
```python
python -m tutoreval.generate --model mistralai/Mistral-7B-Instruct-v0.2 --hf_chat_template
```



The script `./tutoreval/generate.sh` provides some useful utilities for running your model.

### ☑️ Grading outputs with GPT-4
`grade.py` grades the LM tutor outputs and updates `./openbook` and `./closedbook` with the GPT-4 grades.
The script `./tutoreval/grade.sh` also provides some utilities for grading.
