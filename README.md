# Alignment Personalization

- Update: Our paper was accepted to ACL 2025 (main)! 🎉

This is the official repository of "Whose Boat Does it Float? Improving Personalization in Preference Tuning via Inferred User Personas", which can be viewed [here](https://arxiv.org/abs/2501.11549).

<img width="1329" alt="persona_main" src="https://github.com/user-attachments/assets/5bc7aaff-e6c3-4f4a-b967-e017f8ea8613" />

## Table of Contents

- [Abstract](#abstract)
- [Datasets and Pre-trained Models](#datasets-and-pre-trained-models)
- [Setup](#setup)
- [Persona Inference](#persona-inference)
  - [Using LLMs for Persona Inference](#using-llms-for-persona-inference)
  - [Augmenting your Preference Data](#augmenting-your-preference-data)
  - [Evaluation](#evaluation)
    - [Accuracy](#accuracy)
    - [Persona Preferences](#persona-preferences)
    - [Word Saliency](#word-saliency)
- [Persona Tailoring](#persona-tailoring)
  - [Training LLMs for Persona Tailoring](#training-llms-for-persona-tailoring)
  - [Evaluation](#evaluation)
- [Known Limitations](#known-limitations)
- [Contact](#contact)


## Abstract

LLMs are tuned to follow instructions (aligned) by learning which of two outputs users prefer for a prompt. However, this preference data format does not convey *why* users prefer responses that are chosen or rejected, so LLMs trained on these datasets cannot tailor responses to varied user needs. To surface these parameters of personalization, we apply **abductive reasoning** to preference data, inferring needs and interests of users, i.e., personas, that may prefer each output.

We test this idea in two steps:
1. **Persona Inference (PI):** abductively inferring personas of users who prefer chosen or rejected outputs.
2. **Persona Tailoring (PT):** training models to tailor responses to personas from PI.

Our key findings are three-fold:
1. LLMs infer personas accurately explaining why different users may prefer both chosen or rejected outputs.
2. Training on preference data augmented with PI personas via PT boosts personalization, enabling models to support user-written personas.
3. Rejected response personas form harder personalization evaluations, showing PT better aids users with uncommon preferences versus typical alignment methods.

We argue for an abductive view of preferences for personalization, asking not only which response is better but when, why, and for whom.

## Datasets and Pre-trained Models

We provide pre-processed datasets and pre-trained models on Hugging Face: [Alignment Personalization Repository](https://huggingface.co/collections/nbalepur/alignment-personalization-6761ee17403573affaa670aa). Specifically, it contains the following datasets:
- [nbalepur/persona-inference](https://huggingface.co/datasets/nbalepur/persona-inference): Dataset used for Persona Inference experiments.
- [nbalepur/persona-tailoring](https://huggingface.co/datasets/nbalepur/persona-tailoring): Dataset used for Persona Tailoring experiments.

We also provide the following preference-tuned, personalized models:
- [nbalepur/Llama-3.1-8B-PT-DPO-BeaverTails](https://huggingface.co/nbalepur/Llama-3.1-8B-PT-DPO-BeaverTails): Answers advice queries tailored to personas.
- [nbalepur/Llama-3.1-8B-PT-DPO-HHH](https://huggingface.co/nbalepur/Llama-3.1-8B-PT-DPO-HHH): Responds to dialogue requests tailored to personas.
- [nbalepur/Llama-3.1-8B-PT-DPO-Mnemonic](https://huggingface.co/nbalepur/Llama-3.1-8B-PT-DPO-Mnemonice): Generates mnemonic devices tailored to personas.

All models use the following prompt template:
```
Prompt: [input prompt]
Persona: [input persona]
Response:
```

## Setup

Follow these steps to set up the environment:

1. Create a Conda environment:
   ```bash
   conda create -n "persona" python=3.11.0
   conda activate persona
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Fill in the API keys in `.env` depending on which functionality of the repository you want to use:
   ```bash
   HF_READ_TOKEN=... # read Huggingface models
   HF_WRITE_TOKEN=... # write to Huggingface
   OPEN_AI_TOKEN=... # OpenAI token
   COHERE_TOKEN=... # Cohere token
   ANTHROPIC_TOKEN=... # Anthropic token
   WANDB_KEY=... # Weights and Biases token
   ```

## Persona Inference

This section provides the code for **Persona Inference**, which uses LLMs to abductively reason why responses may be preferred in preference datasets. The task is as follows:
- For prompt $p$ and responses $r_1$ and $r_2$, the LLM gives a persona $\mathcal{P}_1\$ such that a user described by $\mathcal{P}_1$ would prefer $r_1$ over $r_2$.

The code for the Persona Inference experiments (Sections 2 and 3 in the paper) is located in `/persona-inference/` and can be run via `model/run_model.py`. The arguments for this Python code are detailed in the following table:

| **Argument**            | **Sample Value**                           | **Description**                                                              |
|--------------------------|-------------------------------------|------------------------------------------------------------------------------|
| `inference_split`        | `BeaverTails`                         | Split of the dataset to use.                                                 |
| `dataset_name`           | `nbalepur/persona-inference`       | Name of the Hugging Face dataset to run PI on.                                            |
| `partition`              | `full`                             | Partition of the dataset to run.                                             |
| `model_type`             | `open_ai`                          | Type of the model (defined in `ModelType` in `enums.py`).                    |
| `model_name`             | `gpt-4o-mini`                      | Endpoint for the model.                                                      |
| `run_name`               | `default`                          | Name used to identify this run.                                              |
| `experiments`            | `("persona_inference")`            | List of experiments to run (defined in `PromptType` in `enums.py`).          |
| `temperature`            | `0.0`                              | Sampling temperature for the model.                                          |
| `min_tokens`             | `5`                                | Minimum number of tokens to generate.                                        |
| `max_tokens`             | `200`                              | Maximum number of tokens to generate.                                        |
| `stop_token`             | `\\nPrompt:`                      | Token used to stop generation.                                               |
| `device_map`             | `auto`                             | Device to load tensors (`'cpu'`, `'cuda'`, `'auto'`).                        |
| `hf_token`               | `hf_...`               | Hugging Face token for accessing datasets/models (specify in .env).                            |
| `open_ai_token`          | `sk-...`               | OpenAI API token for accessing OpenAI models (specify in .env).                                |
| `cohere_token`           | `...`                | Cohere API token for accessing Cohere models (specify in .env).                                |
| `anthropic_token`        | `...`             | Anthropic API token for accessing Anthropic models (specify in .env).                          |
| `res_dir`                | `results/`                         | Directory to save results.                                                   |
| `prompt_dir`             | `prompts/`                         | Directory containing prompts.                                                |
| `cache_dir`              | `""`                               | Directory to cache any models/datasets.                                               |
| `load_in_4bit`           | `--True` (flag)                    | Flag to load models in 4-bit precision.                      |
| `load_in_8bit`           | `--True` (flag)                    | Flag to load models in 8-bit precision.                      |

We provide a sample bash script to help you run the function in `/scripts/example.sh`. The script should be run via:

```bash
bash scripts/example.sh
```

### Using LLMs for Persona Inference

To run Persona Inference with a model, perform the following:
1. Set `run_name` to a string defining what the run should be called.
2. Set `model_name` and `model_type` to the model you want to use for persona inference.
3. Set `experiments=(persona_inference)`.

Then, run:

```bash
bash scripts/example.sh
```

The LLM personas will be stored as a JSON in `/results/`, which can be used to augment a dataset or evaluated using the code below.

### Augmenting your Preference Data

After running persona inference, you can augment your dataset with LLM-inferred personas using the Python code in `model/build_augmented_dataset.py`, which has the following arguments:

| **Argument**            | **Sample Value**                | **Description**                                                      |
|--------------------------|--------------------------|----------------------------------------------------------------------|
| `dataset_name`           | `"nbalepur/persona-inference"`                    | Original Hugging Face dataset path name.                                               |
| `run_name`               | `default`               | Run name used to generate personas.                                                |
| `model`                  | `gpt-4o-mini`           | Model used to generate personas.                                     |
| `inference_split`        | `BeaverTails`           | Inference split used to create personas.                             |
| `new_dataset_name`       | `"nbalepur/persona-inference-augmented"`                    | New name for the Hugging Face dataset after augmentation.                         |
| `hf_write_token`         | `hf_...`   | Hugging Face write token for uploading the new dataset.                 |
| `push_to_hub`         | `--True` (flag)   | Flag to push the dataset to the huggingface hub                |

We again provide a sample bash script at:

```bash
bash scripts/build_data.sh
```

### Evaluation

Below provides the scripts to evaluate the quality of LLM personas.

#### Accuracy

To evaluate the accuracy of any model that has run Persona Inference, perform the following:
1. Set `run_name` to the run used in Persona Inference you would like to evaluate.
2. Set `model_name` and `model_type` to the model you want to use for judging.
3. Set `experiments=(persona_accuracy)`.

Then, run:

```bash
bash scripts/example.sh
```

The code will automatically collect all models that have run Persona Inference using `run_name`.

To plot and compute the accuracy (Figure 3 shown in the paper), you can run:

```python
python evaluation/accuracy_plot.py
```

At the top of the script, you will need to specify the list of models, datasets, and run names you want to compute accuracy for. The accuracy plot will appear in `evaluation/images/accuracy_plot.pdf`.

#### Persona Preferences

For persona preferences evaluation:
1. Set `run_name` to the run used in Persona Inference you would like to evaluate.
2. Set `model_name` and `model_type` to the model you want to use for judging.
3. Set `experiments=(prefs)`.

Then, run:

```bash
bash scripts/example.sh
```

The code will automatically collect all models that have run Persona Inference using `run_name`.

To plot and compute the LLM judge's preferences over personas (Figure 4 shown in the paper), you can run:

```python
python evaluation/persona_prefs_plot.py
```

At the top of the script, you will need to specify the list of models, datasets, and run names you want to compute accuracy for. The persona preferences plot will appear in `evaluation/images/persona_prefs.pdf`.

#### Word Saliency

To find the most salient words in chosen and rejected personas (Table 1 in the paper), you can run:

```python
python evaluation/word_saliency.py
```

At the top of the script, you will need to specify the model/run name used to infer personas, the original Hugging Face dataset used, the split of the dataset used, and the frequency cutoff that determines how popular a word must be to be included in the saliency computation (in the paper, we use 10). The top-10 most salient words in chosen and rejected personas will be printed.

## Persona Tailoring

We provide our code for Persona Tailoring below, which can align a much more personalized model by training on LLM personas from Persona Inference:

- For prompt $p$ and persona $\mathcal{P}$, the LLM gives response $r$ for $p$ that is tailored to $\mathcal{P}$.

All of the code for the Persona Tailoring experiments (Sections 4 and 5 in the paper) can be found in `/persona-tailoring/`. Before running any code, we recommend filling in `/model/config.py` so most parameters are shared across model runs. Specifically, there are the following arguments:

| **Parameter**          | **Sample Value**                          | **Description**                                                     |
|-------------------------|------------------------------------|---------------------------------------------------------------------|
| `model_nickname`       | `llama_1b`                        | Nickname for the model being used.                                  |
| `base_model_name`      | `meta-llama/Llama-3.2-1B`         | Full name of the base model.                                        |
| `use_wandb`            | `True`                            | Whether to use Weights & Biases for experiment tracking.            |
| `dataset_name`         | `BeaverTails`                     | Name of the dataset used.                                           |
| `load_in_8bit`         | `False`                           | Whether to load the model in 8-bit precision.                       |
| `load_in_4bit`         | `False`                           | Whether to load the model in 4-bit precision.                       |
| `cache_dir`            | ``   | Directory for caching models or datasets.                           |
| `model_save_dir`       | `` | Directory to save models after training.                        |
| `results_save_dir`     | `results/`                        | Directory to save experiment results.                               |
| `device_map`           | `auto`                            | Device mapping for loading tensors (`'cpu'`, `'cuda'`, `'auto'`).   |

The rest of the config parameters will be generated automatically or loaded from the `.env` file (API keys/tokens).

### Training LLMs for Persona Tailoring

We implement three strategies for Persona Tailoring: Few-Shot Prompting, Supervised-Fine Tuning, and Direct Preference Optimization, which use the following arguments:

| **Argument**         | **Type** | **Choices**                | **Description**                                    |
|----------------------|--------------|----------------------------|----------------------------------------------------|
| `--training_type`     | `str`    | `TrainingType enum` | Persona type to train on (`chosen`, `rejected`, `all`, `none`). |
| `--inference_type`    | `str`    | `InferenceType enum`| Persona type to run inference on (`gold_chosen`, `gold_rejected`, `retr_chosen`, `retr_rejected`, `system`, `none`). |

The FS model can immediately run inference since it is based on an existing LLM. The SFT and DPO models undergo a three-step process of:
1. Training the model with LoRA.
2. Merging the model weights.
3. Running inference to generate text.

To simplify this, we combine all experimental steps in a single bash script for each technique:

#### Few-Shot Prompting
```bash
bash scripts/few-shot.sh
```

#### Supervised Fine-Tuning (SFT)
```bash
bash scripts/sft.sh
```

#### Direct Preference Optimization (DPO)
```bash
bash scripts/dpo.sh
```

### Evaluation

We use the [Prometheus-2 (7B)](https://github.com/prometheus-eval/prometheus-eval) LLM judge to compare model outputs. It compares models in **Response Quality** and **Personalization**. The judge can be run via `evaluation/run_judge.py` and win-rate can be computed via `evaluation/win_rate.py`. These Python scripts use the arguments below:

| **Argument**              | **Sample Value**            | **Description**                                                      |
|----------------------------|----------------------|----------------------------------------------------------------------|
| `model_type_base (ModelType enum in util.py)`          | `dpo`               | Base model type used for evaluation.                                 |
| `training_type_base (TrainingType enum in util.py)`       | `chosen`            | Training type for the base model.                                    |
| `inference_type_base (InferenceType enum in util.py)`      | `gold-chosen`       | Inference type for the base model.                                   |
| `model_type_test (ModelType enum in util.py)`          | `sft`               | Test model type used for evaluation.                                 |
| `training_type_test (TrainingType enum in util.py)`       | `chosen`            | Training type for the test model.                                    |
| `inference_type_test (InferenceType enum in util.py)`      | `gold-chosen`       | Inference type for the test model.                                   |

We again provide a bash script to do both of these at:

```bash
bash scripts/judge.sh
```

## Known Limitations

We note the following limitations with our approach:
1. Models have not undergone safety training, so they can produce harmful, biased, or irrelevant text when instructed to.
2. The Anthropic HHH models often generate repetitive text. To fix this, we recommend using decoding strategies apart from greedy decoding. In the paper, we filtered out these repetitive outputs during evaluation

## Contact

If you have any questions about the paper or repository, feel free to raise an issue or contact me, Nishant Balepur, at [nbalepur@umd.edu](mailto:nbalepur@umd.edu)
