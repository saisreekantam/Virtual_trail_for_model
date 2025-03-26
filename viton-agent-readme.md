## About VITON-HD

VITON-HD is a high-resolution virtual try-on system that enables realistic clothing transfer onto images of models. It implements the paper "VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization" (CVPR 2021).

The original implementation provides state-of-the-art virtual try-on capabilities at 1024x768 resolution, but requires multiple manual preprocessing steps. Our CrewAI agent automates these steps for a streamlined experience.


## New Features

- **Fully Automated Workflow**: Our CrewAI agent takes care of all preprocessing steps
- **Automatic Cloth Mask Generation**: No need to manually create cloth masks
- **Simplified Testing**: Single function call to process a clothing item
- **Mac Compatibility**: Works on both Intel and Apple Silicon Macs

## Installation

### 1. Clone this repository:

```bash
git clone https://github.com/saisreekantam/Virtual_trail_for_model.git
cd VITON-HD
```

### 2. Create a conda environment:

```bash
conda create -y -n viton_env python=3.8
conda activate viton_env
```

### 3. Install dependencies:

For Intel/AMD machines with CUDA:
```bash
conda install -y pytorch>=1.6.0 torchvision cudatoolkit>=9.2 -c pytorch
pip install opencv-python crewai
```

For Mac (including Apple Silicon):
```bash
conda install -y pytorch>=1.6.0 torchvision -c pytorch
pip install opencv-python crewai
```

### 4. Download datasets and checkpoints

Download the necessary pre-trained model weights (*.pkl) and test images from [my Drive folder]((https://drive.google.com/drive/folders/1Z1FfckL5fKHQO1cMK7NYpDM_q3rdnOxI?usp=drive_link)) (for checkpointes) and [my_Drive_folder]((https://drive.google.com/file/d/1VPavwYP53xMGncwo8mTEvrVgu65RxEU3/view?usp=drive_link)) and unzip the files.

Place the contents in the following directories:
- Model weights: `./checkpoints/`
- Dataset files: `./datasets/`

## Usage

### Using the CrewAI Agent

```python
from viton_agent import VITONHDCrew

# Create the agent
viton_crew = VITONHDCrew()

# Process a cloth image (provide the path to your clothing image and a model ID)
result_path = viton_crew.execute_cloth_processing(
    cloth_image_path="path/to/your/cloth_image.jpg",
    model_image_id="000010_0"  # ID of a model in the VITON-HD dataset
)

print(f"Virtual try-on result saved to: {result_path}")
```

### Using the VITON-HD Agent Directly

For more control over the process:

```python
from viton_agent import VITONHD_Agent

# Initialize the agent
agent = VITONHD_Agent()

# Process a clothing image through the complete pipeline
result_path = agent.process_cloth_image(
    cloth_image_path="path/to/your/cloth_image.jpg", 
    model_image_id="000010_0",
    output_name="my_test_output"
)
```

### Manual Testing with Original Script

You can also use the original VITON-HD testing script:

```bash
python test.py --name test_output
```

## How It Works

The CrewAI agent integrates with VITON-HD to automate these steps:

1. **Cloth Mask Generation**: Creates a binary mask separating the clothing item from its background
2. **Test Pair Creation**: Generates the test_pairs.txt file with model and clothing pairs
3. **Virtual Try-On**: Runs the VITON-HD pipeline with your inputs
4. **Result Retrieval**: Returns the path to the generated image

## Directory Structure

```
viton-hd-agent/
├── checkpoints/         # Store model weights here
├── datasets/            # Dataset files
│   └── test/
│       ├── cloth/       # Input clothing images
│       └── cloth-mask/  # Generated clothing masks
├── results/             # Output images
├── test.py              # Modified test script for Mac compatibility
├── viton_agent.py       # CrewAI agent implementation
└── ... (other VITON-HD files)
```

## Dependencies

- Python 3.8
- PyTorch >= 1.6.0
- OpenCV
- CrewAI
- NumPy

## License

This project extends VITON-HD, which is available under Creative Commons BY-NC 4.0. You can use, redistribute, and adapt the material for non-commercial purposes, as long as you give appropriate credit by citing the original paper.

## Citation

If you find this work useful for your research, please cite the original VITON-HD paper:

```
@inproceedings{choi2021viton,
  title={VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization},
  author={Choi, Seunghwan and Park, Sunghyun and Lee, Minsoo and Choo, Jaegul},
  booktitle={Proc. of the IEEE conference on computer vision and pattern recognition (CVPR)},
  year={2021}
}
```

## Acknowledgements

- [VITON-HD](https://github.com/shadow2496/VITON-HD) - Original implementation
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Framework for building AI agents
