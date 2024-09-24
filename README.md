Here's the updated README incorporating the information from the previous one, now organized and tailored for your project:

---

# CNN Vessel Segmentation Project

This project focuses on interpreting the segmentation of vessels in medical images using Convolutional Neural Networks (CNNs). The goal is to train a CNN model that can accurately predict vessel structures in medical imaging and analyze the model's predictions using gradient-based methods. The project is organized into several key directories and scripts, each serving a specific purpose, from data handling and model training to gradient extraction and validation.

![Watch the example video!](./assets/InterpretingCNNsApp-Overview.gif)

## How to Run the Project

### Prerequisites

- Python 3.8+
- PyTorch
- Numpy
- Matplotlib
- Pillow
- Plotly
- Dash & Dash Bootstrap Components
- Jupyter Notebook (optional, the notebook is used only for visual validation)

_To install the requireds libs use the command:_

```bash
pip install torch numpy matplotlib pillow plotly dash dash-bootstrap-components jupyter
```

### 1. Train the Model

To train the CNN model for vessel segmentation:

- Run the training script:

```bash
python models/train.py
```

This script will initialize the model, load the dataset, and begin training. The trained model weights will be saved in the `trained-models/` directory.

- For an interactive training experience with metrics and visualization, run the following Jupyter notebook:

```bash
jupyter notebook models/vessel_training_evaluation.ipynb
```

The notebook follows the same logic as the training script but includes additional evaluation steps for deeper analysis.

### 2. Extract Gradients

After training the model, you can extract gradients using the following scripts:

1. **Extract Raw Gradients**:

   - Run the gradient extraction script:

     ```bash
     python gradient-extraction/extract_gradients.py
     ```

     This script loads the trained model and extracts the Jacobian matrix for each image in the dataset, saving the gradients in the `gradients/` directory.

2. **Extract Thresholded Gradients**:

   - Run the thresholded gradient extraction script:

     ```bash
     python gradient-extraction/extract_thresholded_gradients.py
     ```

     This script processes the raw gradients to create thresholded images that highlight significant gradient values. These images are saved in the `thresholded_gradients/` directory.

### 3. Validate Gradients(Optional)

To validate the extracted gradients and visualize the results:

- Open the Jupyter notebook and run it with the new gradients generated:

  ```bash
  jupyter notebook gradient-extraction/validate_gradients.ipynb
  ```

### 4. Interpret CNN based on Gradients

- Finnaly, use the interactive Dash application for dynamic visualization of the gradients:

  ```bash
  cd plotly && python plotly/interatively_interpretation_cnn_app.py
  ```

This application allows you to select images, visualize their gradients, and adjust thresholds for detailed gradient analysis.

## Project Organization

- **`data/`**: Contains the dataset used for training and validating the CNN model.

  - `cropped_images/`: Sample images for training and testing.
  - `VessMap/`: The main dataset with images, labels, and skeletons.
  - `vess_map_dataset_loader.py`: Script for loading and preprocessing the dataset using PyTorch's DataLoader.
  - **Tips**: Adapt this directory by updating paths and formats to accommodate different datasets or image types.

- **`models/`**: Contains model architectures, training scripts, and evaluation results.

  - `train.py`: Script to train the custom CNN model.
  - `vess_map_custom_cnn.py`: Defines the simplified ResNet architecture tailored for vessel segmentation.
  - `vessel_training_utils.py`: Utility functions for training, including metrics and visualization.
  - `vessel_training_evaluation.ipynb`: Jupyter notebook for training and evaluating the model interactively.
  - `trained-models/`: Stores the trained model weights.
  - `evaluation-results/`: Holds the evaluation metrics and visual results from training sessions.
  - **Tips**: Modify the model architecture and training parameters to suit different datasets or hardware capabilities.

- **`gradient-extraction/`**: Dedicated to analyzing the gradients of the trained CNN models.

  - `extract_gradients.py`: Extracts all the gradients for a given input by calculating the Jacobian matrix.
  - `extract_thresholded_gradients.py`: Processes extracted gradients to create thresholded gradient images.
  - `validate_gradients.ipynb`: Notebook to validate and visualize the extracted gradients.
  - `gradients/`: Stores the raw gradients as PyTorch tensors.
  - `thresholded_gradients/`: Stores the processed thresholded gradient images.
  - **Tips**: Customize gradient extraction and processing to focus on specific layers or output channels for more detailed analysis.

- **`plotly/`**: Contains the script for the interactive Dash application for visualizing and interpreting CNN gradients.

  - `interatively_interpretation_cnn_app.py`: A Dash application that allows users to interactively visualize gradients and model predictions. This tool provides an intuitive interface for understanding how the model interprets various input features.
  - **Tips**: Use this application to experiment with different images and thresholds to gain insights into the model’s behavior on specific cases.
