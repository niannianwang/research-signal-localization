# Signal Classification Research Project

This repository contains the organized code for signal detection and Direction of Arrival(DoA) research, including CNN and GLRT analysis and adversarial attacks.

## Project Structure

```
research_organized/
├── data/
│   ├── original/                  # Raw .mat signal files
│   └── processed/                 # Improved, filtered datasets
├── models/
│   ├── attack/                    # Adversarial trained models
│   └── noattack/                  # Clean-trained CNN models
├── notebooks/
│   ├── 1_baseline_models.ipynb
│   ├── 2_adversarial_attacks.ipynb
│   ├── 3_data_improvement.ipynb
│   ├── 4_improved_adversarial_attacks.ipynb
│   ├── 5_baselines.ipynb
│   ├── 6_doA.ipynb
│   ├── 7_doa_data_improvement.ipynb
│   ├── 8_doa_adversarial_attacks.ipynb
│   ├── 9_doa_baselines.ipynb
│   └── (and others for individual baseline training/evaluation)
├── results/
│   ├── attacks/                        # Standard attack accuracy results for detection
│   ├── baseline/
│   │   └── doa/                        # DoA-specific baselines
│   └── improved_attacks/
│       └── doa/                        # DoA-specific attack accuracy
│   ├── noattack/                       # Clean GLRT prediction and accuracy
├── src/
│   ├── adversarial_attacks.py
│   ├── baselines.py
│   ├── cnn.py
│   ├── data_improvement.py
│   ├── doa.py
│   ├── glrt.py
│   └── preprocessing.py
└── README.md
```

## Usage

1. Install Dependencies:
   *(Install list to be updated later — placeholder)*  
   ```bash
   pip install -r requirements.txt
   ```

2. Place your original dataset (`processed_signals.mat`) in the `data/original/` directory.

3. Run the notebooks in order:

   Signal Detection (Binary)
   - `1_baseline_models.ipynb`: Trains and evaluates the baseline CNN and GLRT models  
   - `2_adversarial_attacks.ipynb`: Runs FGSM and PGD attacks on the original dataset  
   - `3_data_improvement.ipynb`: Improves the dataset by removing misclassified samples  
   - `4_improved_data_attacks.ipynb`: Runs attacks on the improved dataset  
   - Baselines Training
     - `adversarial_train.ipynb`: Train CNN with adversarial samples  
     - `defense_distillation.ipynb`: Train CNN using defense distillation  
   - `5_baselines.ipynb`: Combines CNN, GLRT, and baseline models against adversarial attacks  

   Direction of Arrival (DoA)
   - `6_doa.ipynb`: Train and evaluate the baseline CNN on GLRT models  
   - `7_doa_data_improvement.ipynb`: Improves the dataset by removing misclassified samples  
   - `8_doa_adversarial_attacks.ipynb`: Runs FGSM and PGD attacks on the original dataset  
   - Baselines Training
     - `doa_adversarial_train.ipynb`: Train CNN with adversarial samples  
     - `doa_defense_distillation.ipynb`: Train CNN model using defense distillation  
   - `9_doa_baselines.ipynb`: Combines CNN, GLRT, and baseline models against adversarial attacks

4. Results and visualizations will be saved in the `results/` directory.

## Key Features

- Modular code organization with separate files for different components
- Automatic model saving and loading
- Comprehensive adversarial attack analysis (FGSM and PGD with L-inf and L-2 norms)
- Data improvement strategy with balanced removal of misclassified samples
- Results saving and visualization for easy comparison

## Notes

- Models are automatically saved after training and can be loaded for later use
- Results are saved in csv files for easy access and analysis
- The improved dataset is saved as a new .mat file for detection and .npz file for DoA 