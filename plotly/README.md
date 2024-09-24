# Plotly - Interactively Interpreting CNNs

This directory contains a Dash application that allows for interactive interpretation of Convolutional Neural Networks (CNNs) by visualizing and analyzing gradients for vessel segmentation tasks in medical images.

## Overview

The `interatively_interpretation_cnn_app.py` script is a web-based interactive tool that visualizes CNN model outputs and their corresponding gradients. The application helps users understand which parts of an input image influence the model's predictions the most by focusing on significant gradient regions.

## Key Features

- **Interactive Image Selection**: Users can select images from the dataset to analyze.
- **Gradient Visualization**: The application plots the gradients of selected images, highlighting areas with significant gradient values.
- **Bounding Box Display**: The tool draws bounding boxes around the most significant gradient areas, helping users quickly identify important regions.
- **Custom Thresholding**: Users can adjust the threshold to filter out less significant gradients, focusing on the most influential pixels.
- **Model Mask Overlay**: Visualize the model's predicted segmentation mask alongside the gradients.

## How to Use

### Running the Application

To start the Dash application, run the following command:

```bash
python interatively_interpretation_cnn_app.py
```

After running the script, visit `http://127.0.0.1:8050/` in your web browser to interact with the application.

### Folder Structure

- `interatively_interpretation_cnn_app.py`: The main script that runs the Dash application.
- `../gradient-extraction/gradients/`: Directory where the gradients are stored as `.pt` files. These gradients are used for visualization.
- `../gradient-extraction/thresholded_gradients/`: Contains thresholded gradient images that highlight significant gradient areas.
- `../models/trained-models/vess_map_custom_cnn.pth`: The path to the pre-trained CNN model used in this application.
- `../data/cropped_images/`: Directory containing the input images used in the application.

### Example Usage

- **Select Image**: Choose an image from the dropdown menu to analyze its gradients.
- **View Gradients**: The app will display the original image, significant gradient pixels, and the model's prediction mask. Browse through the images and it's pixels to see the gradients values.
- **Adjust Threshold**: Modify the threshold value to change the sensitivity of the gradient display.

## Customization

You can customize the application by adjusting paths, modifying the gradient extraction logic, or changing the visualization parameters. Below are some areas where you might want to adapt the script:

- **Model Architecture**: If you're using a different CNN model, update the model loading section in `load_model`.
- **Gradient Calculation**: Modify the gradient extraction method if you want to experiment with different gradient-based interpretations.
- **User Interface**: Customize the Dash components for a better user experience, such as adding more interactivity or different visualizations.
