# ATLANTIS: A Framework for Automated Targeted Language-guided Augmentation Training for Robust Image Search

## Overview
ATLANTIS is a framework designed to enhance the generalizability of Deep Metric Learning (DML) models in Image Search Applications based on the content-based image retrieval (CBIR) systems through automated and targeted synthetic data augmentation. The framework comprises multiple components that collectively address training data deficiencies and enhance model robustness in various data scenarios.

## Repository Structure
This repository is organized into several key components and scripts:

1. **Targeted Synthetic Data Generation: `scripts/data_generation/`**
    - `data_insight_generator.py`: Contains functions for extracting contextual insights and identifying deficiencies in the existing training data with the hybrid approach
    - `data_insight_generator_llm_only.py`: Contains functions for extracting contextual insights and identifying deficiencies in the existing training data with LLM-only approach
    - `data_insight_generator_heuristic.py`: Contains functions for extracting contextual insights and identifying deficiencies in the existing training data with heuristic approach
    - `caption_generator.py`: Generates enhanced captions for the training dataset.
    - `missing_caption_generator.py`: Fills in missing captions for training and test datasets.
    - `aps_no_feedback.py`: Augmentation protocol selector script for generating target synthetic data description objectives
    - `image_to_text.py`: Transforms available image-space information to text-space for the language guided synthetic data generation pipeline 
    - `texttoimage.py`: Utilizes a text-to-image model to generate synthetic images based on the enhanced captions.

2. **Dataset Preparation and Management: `scripts/data_processing`**
    - `restructure_cars_dataset_with_labels.py`: Restructures the dataset and organizes images into class directories.
    - `outlier_remover.py`: Detects and removes outliers from the synthetic data.
    - `data_creater.py`: Merges synthetic and original datasets to create a comprehensive training set.
    - `raw_data_info.py`: Analyzes and provides information on the raw data distribution.
    - `json_cleaner.py`: Cleans and converts the generated captions into a list format.
    - `json_checker.py`: Checks and validates the generated JSON data.
    - `json_merger.py`: merges JSON data.


3. **Training Image Search Models: `scripts/model_training`**
    - `train_vits.py`: Trains the ViT-S model on the augmented dataset.
    - `train_vits_allshot.py`: Trains the ViT-S model on all-shot data.
    - `eval.py`: For evaluating test data performance.

## Installation
To set up the ATLANTIS framework, follow these steps:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/intherejeet/ATLANTIS.git
    cd ATLANTIS
    ```

2. **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ``` 
    (will be updated soon)

3. **Configure API Keys:**
    Update the OpenAI API configuration in the relevant scripts with your API keys and endpoint information.

## Key Components

1. **Data Insight Generator (DIG):** 
    Extracts contextual insights and deficiencies from the existing training data using visual and metadata analysis.

2. **Augmentation Protocol Selector (APS):** 
    Defines dynamic, context-aware augmentation strategies based on the identified data insights.

3. **Outlier Removal and Diversity Control (ORDC):** 
    Ensures the semantic coherence and diversity of the synthetic data by removing outliers.

## Results and Evaluation
ATLANTIS has demonstrated significant performance improvements in domain-scarce, class-imbalanced, and adversarial scenarios. The comprehensive empirical evaluations reveal that ATLANTIS surpasses state-of-the-art methods, establishing it as a robust and scalable framework for CBIR.

## Contributing
We welcome contributions to enhance ATLANTIS. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the terms of the CC-BY license.


For more detailed information and the empirical evaluations, please refer to the [ATLANTIS BMVC 2024 Paper](./ATLANTIS_FRE_BGU_BMVC2024_Camera_Ready.pdf).
