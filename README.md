# Low-Light Waste Classification

Group term project for the Computer Vision course (BBM418 / AIN433) at Hacettepe University.

We built a 4-class waste classifier (glass, metal, paper, plastic) and studied how it behaves when images are taken in poor lighting.

## What we did

1. **Baseline models**: trained and compared a custom CNN, VGG16 and ResNet50 on the Kaggle garbage classification dataset.
2. **Low-light test set**: created a darkened test set by applying brightness and color jitter to the original images.
3. **Enhancement network**: designed a small CNN that brightens and restores an image before it is classified. Its loss focuses on keeping object structure, not just matching colors, so the classifier still recognises the object after enhancement.
4. **Experiments**: tried different loss weightings, stratified sampling and focal loss for the classifier on enhanced images.

## Files

| File | Contents |
|---|---|
| `01_baseline_models.ipynb` | Data preparation, custom CNN / VGG16 / ResNet50 training and comparison |
| `02_enhancer_pipeline.ipynb` | Low-light transform, enhancement network and classification on enhanced images |
| `results/` | Confusion matrices of the baseline models |

## Running

The notebooks were run on Google Colab. Download a garbage classification dataset from Kaggle and point the dataset path in the notebook to it.

```bash
pip install numpy opencv-python matplotlib seaborn torch torchvision pillow tqdm scikit-learn
```

## Tech

Python, PyTorch, torchvision, OpenCV, scikit-learn
