# MLP_Group_Project

# 1. How to Train a Model
### 1. Install requirements (see Section 4)

### 2. Specify your parameters in a yaml file in the folder `yamls`.

You can for example specify where to store the results and weights of your model (under `destionation_path`) or how many epochs you want to have.


### 3. cd into the `scripts` folder 
### 4. Run the model: 
```
python run.py ../yamls/<your_yaml.yaml>
```
for example 
```
python run.py ../yamls/base.yaml
```
### 4. For debugging:
You can use the smaller debug datasets (`data/hatespeech/debug_train.csv`), so you don't have to wait too long ;)

# 1. How to Test a Model
### 1. Install requirements (see Section 4)

### 2. Specify your parameters in a yaml file in the folder `yamls`.

You can for example specify where to store the results and weights of your model (under `destionation_path`) or how many epochs you want to have. Importantly, you need to specify `load_from` in the embeddings section - this is the path to the checkpoint that you want to load your model from. 
For example:
```
classifier:

  embeddings:
    load_from: "../results/debug2/model_best_acc.pt"
    embeddings: "roberta-base"                           # roberta-base or bert-base-cased
    tokenizer: "roberta-base"                            # roberta-base or bert-base-cased
```
Make sure you specify not only the folder but the path including `.pt`!

### 3. cd into the `scripts` folder 
### 4. Run the model: 
Add the keyword `test` to the end of the command!
```
python run.py ../yamls/<your_yaml.yaml> test
```
for example 
```
python run.py ../yamls/base.yaml test
```


# 3. Directory Structure
```
├── data
│   └── hatespeech
│       └── a_original
│           ├── all_files
│           ├── sampled_test_original
│           └── sampled_train_original
│       ├── debug_dev.csv
│       ├── debug_test.csv
│       ├── debug_train.csv
│       ├── dev.csv
│       ├── test.csv
│       └── train.csv
├── README.md
├── requirements.txt
├── results
├── scripts
│   ├── checkpoints.py
│   ├── classifiers.py
│   ├── datasets.py
│   ├── evaluation.py
│   ├── load_safe_metrics.py
│   ├── model.py
│   ├── plot_loss.py
│   ├── run.py
│   └── training.py
├── split_data.py
└── yamls
    ├── base.yaml
    └── debug.yaml
```

# 4. Requirements
 We recommend to use a virtual environment. 
Here an instruction to install all depencencies with conda.

Create virtual environment with conda called `inter` (short for interpretability) with the requirements installed:

```
conda create --name inter --file requirements.txt
```

Activate the environment:
```
conda activate inter
```

# 5. Dataset
Number of examples:
- Train: 1914 examples

- Dev: 239 examples 

- Test: 239 examples (in the original dataset: 239 dev and test together is their test set)

The data is stored in `data/hatespeech` in csv files (`train/dev/test.csv`). 

The label is separated by a tab (\t) from the example.

# 6. More Resources

Overleaf: https://de.overleaf.com/project/63c8fe7f757d22009c9a1c48