## 🧑‍💻 Evaluating with TutorEval [work in progress]

### Requirements

Please install the following packages:

```python
pip install torch flash_attn transformers accelerate bitsandbytes datasets pandas openai rich
```

### ✍️ Generating LM tutor outputs

`generate.py` constructs the LM tutor outputs for each question and saves them under `./openbook`, or `./closedbook` for TutorEval-ClosedBook.

The script `./tutoreval/generate.sh` provides some useful utilities for running your model.

### ☑️ Grading outputs with GPT-4
`grade.py` grades the LM tutor outputs and updates `./openbook` and `./closedbook` with the GPT-4 grades.
The script `./tutoreval/grade.sh` also provides some utilities for grading.
