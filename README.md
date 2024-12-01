# Sheet-Music-Parser 
This project is designed for training and evaluating deep learning models for music symbol detection and classification using PyTorch. Below, you'll find the instructions to set up the environment, preprocess the dataset, and train the models.


## Getting Started 

### Table of Contents
1. [Setup Environment](#setup-environment)
2. [Dataset Preparation](#dataset-preparation)
   - [Extract Dataset](#extract-dataset)
   - [Split Dataset](#split-dataset)
3. [Running the Model](#running-the-model)
4. [Notes](#notes)


## Setup Environment

### Prerequisites
Ensure you have Conda installed on your machine. If not, download it from [Conda's official site](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).


### Steps to Set Up the Environment
1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/CostanzaSiniscalchi/Sheet-Music-Parser.git
   cd Sheet-Music-Parser
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -n my_env python=3.9 -y
   conda activate my_env
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

### 1. Extract Dataset

There are multiple dataset the can be downlaoaded to train the model on 

- Audiveris Dataset

   This dataset contains 800 of Typeset music sheet images with XML annotations. This data can be used for both Symbol Classification and Object Detection tasks.

   Run the `AudiverisOmrImageExtractor.py` script to extract the dataset into the proper directory: 
   ```bash
   python3 ModelTrainer/datasets/AudiverisOmrImageExtractor.py
   ```

- Fornes Dataset

   This dataset contains 4100 of Handwritten music sheet images. This data can be used for Symbol Classification task.

   Run the `FornesMusicSymbolsImagePreparer.py` script to extract the dataset into the proper directory: 
   ```bash
   python3 ModelTrainer/datasets/FornesMusicSymbolsImagePreparer.py
   ```

- MUSCIMA++ Dataset

   This dataset contains > 90000 annotatations of Handwritten music sheet images, Measure Annotations, MuNG. This data can be used for the following tasks: Symbol Classification, Object Detection, End-To-End Recognition, Measure Recognition.

   Run the `MuscimaPlusPlusImageGenerator2.py` script to extract the dataset into the proper directory: 
   ```bash
   python3 ModelTrainer/datasets/MuscimaPlusPlusImageGenerator2.py
   ```

- OpenOMR Dataset

   This dataset contains 1000 score images of Typeset music sheets. This data can be used for Symbol Classification task.

   Run the `OpenOmrImagePreparer.py` script to extract the dataset into the proper directory: 
   ```bash
   python3 ModelTrainer/datasets/OpenOmrImagePreparer.py
   ```

### 2. Split Dataset
Split the extracted dataset into Training, Validation, and Test sets using the provided script:

   ```bash
   python3 ModelTrainer/datasets/DatasetSplitter.py
   ```

   This will organize the dataset into:

   ```bash
    data/images/
                training/
                validation/
                test/
   ```

## Running the Model
Train the selected model using the TrainModel.py script.
   ```bash
    python3 ModelTrainer/TrainModel.py --dataset_directory dataset_directory --model_name model_name
   ```
- Replace dataset_directory with the path to the dataset directory containing the training, validation, and test folders.
- Replace model_name with the desired model name (resnet or vgg).

Example
   ```bash
    python3 ModelTrainer/TrainModel.py --dataset_directory data/data --model_name resnet
   ```

