# Comparative Diagnostics effect on species classification in POC

This repo is a clone of the original git repository for the baseline paper *Surely Large Multimodal Models (Don't) Excel in Visual Species Recognition?*. The base structure of the code is provided by their research. The original `README.md` for the project has been moved to `ORGINAL_README.md` for reference. 

Our goal was to test if an additional LMM inference step adding a comparative diagnostic would increase the accuracy of the POC model. 

## Directory Structure

Since we are using the existing repository as a base, our directory structure does not reflect exactly what would be expected. Since the `src` directory is required to have source code in it, we moved the entire original root directory into it. Otherwise we would have been making unnecessary changes, which would have increased the chance we do not faithfully recreate the baseline model. Therefore our directory structure is as follows:

```
├── data # Script to download and link to SRC 
├── experiments # The bash script that will run the baseline and comparative diagnostic tests
├── LICENSE # The original license of the PoC code base
├── quen_requirements.txt # Requirements for the qwen python virtual env
├── README.md # Our README
├── requirements.txt # Requirements for the poc python virtual env
├── results # The output results of our testing 
└── src # Contains the root of the original project with our modifications
    ├── assets # Banner image 
    ├── CLIP.md
    ├── config.yml
    ├── data
    ├── dataset_preparation
    ├── DATASETS.md
    ├── dinov3
    ├── finer_topk.py
    ├── lmm-inference
    ├── main.py
    ├── main_ssl.py
    ├── ORGINAL_README.md # Original README from the PoC repository
    ├── output # Output of running our test script
    ├── QUERYLMM.md # Original descriptions of how to use LLM queries
    ├── scripts
    ├── testing.py
    └── utils
```

## Setting up the Environment

There is a guide to set up the environment in `ORGINAL_README.md`, but due to some incompleteness in the details, we have added our environment set up here. 

### CUDA Drivers

You should be using CUDA Drivers version 12.8 as established in the baseline paper. Given configuring drivers varies by platform, we leave this to you. 

### Conda Setup

The first step to set up will be to install Conda, which you can do by following this guide. We used Miniconda, but a full Anaconda Distribution is also a valid approach. You can find the install process [here](https://www.anaconda.com/download/success?reg=skipped) if you do not already have Conda. 

### Conda / Python virtual environments

The following commands should be run in the repository root. 

Create the base conda environment for POC. 

```bash 
# Create base poc conda ENV
conda create -n poc python=3.10 -y
conda activate poc
conda install pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then you will need to create a python virtual environment for poc. Ensure you are actively in the POC conda environment when doing this.

```bash 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Second, create an additional conda environment for using Qwen. 

For LMM inference with Qwen, you can follow [instructions](https://github.com/QwenLM/Qwen2.5-VL) or steps below to set up Qwen2.5-VL-7B locally.

```bash
# setup Qwen2.5-VL-7B locally using huggingface transformers
conda create --name qwen --clone poc
conda activate qwen
```

In the qwen conda environment, create a python virtual environment and install requirements. 

```bash
python3 -m venv .venv_qwen
source .venv_qwen/bin/activate
pip install transformers==4.51.3 accelerate
pip install qwen-vl-utils[decord]
```

### Downloading data

Due to the competition organizers' data having issues with images being inaccessible from the species196 dataset, we were unable to install those and their provided script should not be used. 

We solely experimented with the Semi Aves data set from the [Semi-Supervised Recognition Challenge](https://www.kaggle.com/competitions/semi-inat-2020/data). The only way to get the data is directly from Kaggle. To do this, you will need to set up a [Kaggle API key](https://www.kaggle.com/docs/api#authentication) and Kaggle account.

When downloaded from Kaggle Hub, there is an extra folder level (e.g. `semi-aves/u_train_out/u_train_out` instead of `semi-aves/u_train_out`). To accommodate for this, we create symbolic links.

To download the Aves Dataset run the following commands. 
```bash
conda activate poc
source ./.venv/bin/activate
python3 ./data/download_aves.py
```

This will download the dataset to the top level `./data` folder and create symbolic links in the `./src/dataset` folder.

NOTE: This can take up to 30 minutes. The dataset is about 14GB and needs to be unpacked. We also do not recommend using `ln -s` to create symbolic links because there is so much data in the dataset, it overwhelms the max number of files that can be transferred.


To execute the entire end-to-end pipeline manually instead of using a single script, you can run the following commands sequentially. All commands should be run from inside the `src` directory.

```bash
cd src
```
### Running ONLY the Baseline POC Pipeline

If you want to run the original POC pipeline without the new comparative diagnostic layer, you can run these steps:

#### Step 1: Linear Probing (Expert Initialization)

Start in the root project directory.

```bash
conda activate poc
source .venv/bin/activate
cd src
bash scripts/run_dataset_seed_probing.sh semi-aves 1
```

#### Step 2: Few-shot Finetuning
```bash
bash scripts/run_dataset_seed_fewshot_finetune.sh semi-aves 1
```

#### Step 3: Top-K Prediction Extraction
```bash
bash scripts/run_dataset_seed_topk_fewshot_finetune.sh semi-aves 1
```

#### Step 4: Pre-generate Reference Images
```bash
conda activate qwen
source ../.venv_qwen/bin/activate
cd lmm-inference

python pregenerate_reference_images.py \
    --class-json ../data/semi-aves/semi-aves_labels.json \
    --k 4 \
    --dataset semi-aves \
    --seed-file ../data/semi-aves/fewshot4_seed1.txt \
    --image-root ../data/semi-aves \
    --output-dir ../data/semi-aves
```

#### Step 5: Baseline POC Inference
```bash
python run_inference_local_hf.py \
    --prompt-template top5-multimodal-4shot-with-confidence_ranking \
    --prompt-dir semi-aves \
    --backend huggingface \
    --hf-model-name-or-path Qwen/Qwen2.5-VL-7B-Instruct \
    --config-yaml ../config.yml \
    --image-dir semi-aves \
    --image-paths ../data/semi-aves/test.txt \
    --ref-image-dir ../data/semi-aves/pregenerated_references_4shot \
    --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
    --topk-json ../output/predictions_topk_finetune/finetune_vitb32_openclip_laion400m_semi-aves_4_1_topk_test_predictions.json \
    --output-csv ../output/lmm_results/baseline_poc_semi-aves_4shot_seed1.csv \
    --error-file ../output/lmm_results/baseline_poc_semi-aves_4shot_seed1_errors.txt \
    --max_new_tokens 900
```

#### Step 6: Evaluate Baseline Results
```bash
python eval_output.py \
    --output-csv ../output/lmm_results/baseline_poc_semi-aves_4shot_seed1.csv \
    --test-list ../data/semi-aves/test.txt \
    --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
    --topk-json ../output/predictions_topk_finetune/finetune_vitb32_openclip_laion400m_semi-aves_4_1_topk_test_predictions.json \
    --image-dir semi-aves \
    --config-yaml ../config.yml
```

---

### Running the Baseline AND Comparative Diagnostic Pipelines

#### Step 1: Linear Probing (Expert Initialization)

Start in the root project directory.

```bash
conda activate poc
source .venv/bin/activate
cd src
bash scripts/run_dataset_seed_probing.sh semi-aves 1
```

#### Step 2: Few-shot Finetuning
```bash
bash scripts/run_dataset_seed_fewshot_finetune.sh semi-aves 1
```

#### Step 3: Top-K Prediction Extraction
```bash
bash scripts/run_dataset_seed_topk_fewshot_finetune.sh semi-aves 1
```

#### Step 4: Pre-generate Reference Images
```bash
conda activate qwen
source ../.venv_qwen/bin/activate
cd lmm-inference

python pregenerate_reference_images.py \
    --class-json ../data/semi-aves/semi-aves_labels.json \
    --k 4 \
    --dataset semi-aves \
    --seed-file ../data/semi-aves/fewshot4_seed1.txt \
    --image-root ../data/semi-aves \
    --output-dir ../data/semi-aves
```

#### Step 5: Baseline POC
This step generates standard ranking predictions for comparison.
```bash
python run_inference_local_hf.py \
    --prompt-template top5-multimodal-4shot-with-confidence_ranking \
    --prompt-dir semi-aves \
    --backend huggingface \
    --hf-model-name-or-path Qwen/Qwen2.5-VL-7B-Instruct \
    --config-yaml ../config.yml \
    --image-dir semi-aves \
    --image-paths ../data/semi-aves/test.txt \
    --ref-image-dir ../data/semi-aves/pregenerated_references_4shot \
    --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
    --topk-json ../output/predictions_topk_finetune/finetune_vitb32_openclip_laion400m_semi-aves_4_1_topk_test_predictions.json \
    --output-csv ../output/lmm_results/baseline_poc_semi-aves_4shot_seed1.csv \
    --error-file ../output/lmm_results/baseline_poc_semi-aves_4shot_seed1_errors.txt \
    --max_new_tokens 900
```

#### Step 6: Comparative Diagnostic Generation
This step represents our novel middle LMM layer, generating species-specific comparative diagnostics.
```bash
python run_inference_local_hf.py \
    --prompt-template comparative-diagnostic \
    --prompt-dir semi-aves \
    --backend huggingface \
    --hf-model-name-or-path Qwen/Qwen2.5-VL-7B-Instruct \
    --config-yaml ../config.yml \
    --image-dir semi-aves \
    --image-paths ../data/semi-aves/test.txt \
    --ref-image-dir ../data/semi-aves/pregenerated_references_4shot \
    --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
    --topk-json ../output/predictions_topk_finetune/finetune_vitb32_openclip_laion400m_semi-aves_4_1_topk_test_predictions.json \
    --output-csv ../output/lmm_results/diagnostic_semi-aves_4shot_seed1.csv \
    --error-file ../output/lmm_results/diagnostic_semi-aves_4shot_seed1_errors.txt \
    --max_new_tokens 300
```

#### Step 7: Parse Diagnostics to JSON
```bash
python csv_to_diagnostics_json.py \
    --csv ../output/lmm_results/diagnostic_semi-aves_4shot_seed1.csv \
    --out ../output/lmm_results/diagnostic_semi-aves_4shot_seed1.json
```

#### Step 8: Final POC + Diagnostics
This runs the final POC ranking, but incorporates the diagnostics generated in Step 6.
```bash
python run_inference_local_hf.py \
    --prompt-template top5-multimodal-with-diagnostic_ranking \
    --prompt-dir semi-aves \
    --backend huggingface \
    --hf-model-name-or-path Qwen/Qwen2.5-VL-7B-Instruct \
    --config-yaml ../config.yml \
    --image-dir semi-aves \
    --image-paths ../data/semi-aves/test.txt \
    --ref-image-dir ../data/semi-aves/pregenerated_references_4shot \
    --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
    --topk-json ../output/predictions_topk_finetune/finetune_vitb32_openclip_laion400m_semi-aves_4_1_topk_test_predictions.json \
    --diagnostics-json ../output/lmm_results/diagnostic_semi-aves_4shot_seed1.json \
    --output-csv ../output/lmm_results/poc_with_diagnostic_semi-aves_4shot_seed1.csv \
    --error-file ../output/lmm_results/poc_with_diagnostic_semi-aves_4shot_seed1_errors.txt \
    --max_new_tokens 900
```

#### Step 9: Evaluate Results
Evaluate both the Baseline and the Diagnostic-enhanced outputs to compare accuracy.
```bash
# Evaluate Baseline POC
python eval_output.py \
    --output-csv ../output/lmm_results/baseline_poc_semi-aves_4shot_seed1.csv \
    --test-list ../data/semi-aves/test.txt \
    --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
    --topk-json ../output/predictions_topk_finetune/finetune_vitb32_openclip_laion400m_semi-aves_4_1_topk_test_predictions.json \
    --image-dir semi-aves \
    --config-yaml ../config.yml

# Evaluate POC + Comparative Diagnostic
python eval_output.py \
    --output-csv ../output/lmm_results/poc_with_diagnostic_semi-aves_4shot_seed1.csv \
    --test-list ../data/semi-aves/test.txt \
    --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
    --topk-json ../output/predictions_topk_finetune/finetune_vitb32_openclip_laion400m_semi-aves_4_1_topk_test_predictions.json \
    --image-dir semi-aves \
    --config-yaml ../config.yml
```

