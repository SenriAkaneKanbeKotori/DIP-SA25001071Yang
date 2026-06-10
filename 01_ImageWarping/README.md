# Assignment 1 - Image Warping

This repository is a complete implementation of Assignment 01 for Digital Image
Processing: basic image geometric transformation and point-guided image
deformation.

The implementation follows the provided `README.md` and
`Homework_01 Image Warping` slides:

- Basic image geometric transformation: scale, rotation, translation, and
  horizontal flipping.
- Point-based image deformation with interactive source/target control points.
- Gradio interfaces for real-time parameter tuning and visual checking.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- run_global_transform.py
|-- run_point_transform.py
`-- pics
    |-- teaser.png
    |-- global_demo.gif
    `-- point_demo.gif
```

## Environment

Python 3.9+ is recommended. Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

The code only needs NumPy, OpenCV, and Gradio. A PyTorch environment is fine but
not required for this assignment.

## Run Basic Geometric Transformation

```bash
python run_global_transform.py
```

Upload an image and adjust:

- scale
- rotation
- translation in x/y
- horizontal flip

The transformation is composed around the padded image center and rendered with
OpenCV affine warping.

## Run Point-Guided Deformation

```bash
python run_point_transform.py
```

Click points in alternating order:

1. source point
2. target point
3. next source point
4. next target point

Then click **Run Warping**. Source points are marked blue, target points are
marked red, and green arrows show the requested motion.

The deformation uses inverse affine Moving Least Squares (MLS) sampling. Inverse
sampling avoids holes in the output image. When fewer than three complete point
pairs are selected, the implementation falls back to weighted displacement
interpolation.

## Results

### Basic Transformation

<img src="pics/global_demo.gif" alt="Global transform demo" width="800">

### Point-Guided Deformation

<img src="pics/point_demo.gif" alt="Point-guided deformation demo" width="800">

## References

- Image Deformation Using Moving Least Squares, Schaefer et al.
- Image Warping by Radial Basis Functions, Arad and Reisfeld
- OpenCV Geometric Transformations
- Gradio
