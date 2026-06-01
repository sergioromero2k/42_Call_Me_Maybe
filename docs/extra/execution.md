# 42 Call Me Maybe - Execution and Grading Guide

This document details the steps required to set up the environment, process the private exam exercises, and run the school's automatic grader from scratch.

---

## Step-by-Step Guide

### Step 1: Unzip the project

```bash
unzip moulinette.zip
```

---

### Step 2: Prepare the private exercises

The Moulinette contains a set of hidden functions and tests for evaluation (`--set private`). Run the preparator to clean the directory and generate the exam prompts:

```bash
cd moulinette
uv sync
uv run python -m moulinette prepare_exercises --set private
```

---

### Step 3: Sync data to the project root

Copy the generated questions to your main `data/` folder so the inference pipeline can locate the input files correctly:

```bash
cd ..
cp -r moulinette/data/input/* data/input/
```

---

### Step 4: Run the inference pipeline

Run your pipeline against the exam set. The script reads the prompts from `data/input/` and writes the formatted answer JSON to the output:

```bash
uv run python -m src
```

---

### Step 5: Run the grader

Go back into the Moulinette folder and run the automatic grader pointing to your results file:

```bash
cd moulinette
uv run python moulinette/__main__.py grade_student_answers \
  --set private \
  --student_answer_path ../data/output/function_calling_results.json
```

---

## Expected Output

If all steps were followed correctly, the terminal will display a breakdown of each test result, ending with:

```
+============================================+
|                                            |
|                    PASSED                  |
|            SCORE: 11/11 (100.0%)           |
|                                            |
+============================================+
```