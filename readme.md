# Self-Modifying Neural Network for ARC Challenge

## Project Overview

This project implements a self-modifying neural network architecture designed to tackle the Abstraction and Reasoning Corpus (ARC) challenge. The network can adaptively modify its own structure and hyperparameters based on its performance, aiming to improve its ability to solve diverse reasoning tasks.

### Key Features

- Self-modifying neural network architecture
- Adaptive modification frequency based on performance metrics
- Intelligent decision-making for architectural changes
- Ablation studies to analyze the impact of different components
- Integration with Weights & Biases (wandb) for experiment tracking

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (optional, but recommended for faster training)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/self-modifying-nn-arc.git
   cd self-modifying-nn-arc
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Prepare the ARC dataset:
   - Download the ARC dataset from [the official repository](https://github.com/fchollet/ARC)
   - Place the JSON files in the `arc_challenge` directory

2. Configure the experiment:
   - Open `ablation_study.py`
   - Adjust the `base_config` dictionary to set the initial model configuration
   - Modify the `features_to_ablate` list to specify which features to study

3. Run the experiment:
   ```
   python ablation_study.py
   ```

4. Monitor the progress:
   - The script will print detailed logs to the console
   - Open the Weights & Biases dashboard to view real-time metrics and visualizations

## Project Structure

- `ego.py`: Contains the implementation of the SelfModifyingNetwork and related classes
- `ablation_study.py`: Main script for running experiments and ablation studies
- `data_utils.py`: Utilities for loading and processing the ARC dataset
- `utils.py`: General utility functions

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The ARC challenge creators for providing a benchmark for abstract reasoning in AI
- The PyTorch team for their excellent deep learning framework
- Weights & Biases for their experiment tracking tools

## Contact

For any questions or feedback, please open an issue in this repository.