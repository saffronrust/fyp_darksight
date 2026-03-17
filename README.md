# fyp_darksight
An offensive data preprocessing tool to poison and disrupt Vision Language Models using Nightshade. For my final year project.

## Install dependencies:
`pip install -r requirements.txt`
Note that you will also need to install PyTorch, head to [this link](https://pytorch.org/get-started/locally/) to install PyTorch according to your system's requirements.

## Test Dataset Used
[512 x 512 Dogs Images on Kaggle](https://www.kaggle.com/datasets/greg115/dogs-big)

### Preparing the dataset to be finetuned:
`python prepare_split.py --data_json cap.json --images_root /to_be_trained_on_VLM --out_dir splits`

### Finetune the VLM
`python train_qwen2vl_lora.py`

### Prompt the VLMs to output text descriptions based on the test dataset
`python output_clean_VLM.py`
`python output_poisoned_VLM.py`

### Evaluate the two VLMs' performance
`python evaluate_VLM.py`

