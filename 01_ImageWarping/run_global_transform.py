import cv2
import gradio as gr
import numpy as np


WHITE = (255, 255, 255)


def to_3x3(affine_matrix):
    """Convert a 2x3 affine matrix to homogeneous 3x3 form."""
    return np.vstack([affine_matrix, [0.0, 0.0, 1.0]])


def translation_matrix(tx, ty):
    return np.array(
        [
            [1.0, 0.0, float(tx)],
            [0.0, 1.0, float(ty)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def horizontal_flip_matrix(cx, cy):
    return np.array(
        [
            [-1.0, 0.0, 2.0 * cx],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def normalize_image(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def pad_image(image):
    pad_size = min(image.shape[0], image.shape[1]) // 2
    padded = np.full(
        (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3),
        WHITE,
        dtype=np.uint8,
    )
    padded[
        pad_size : pad_size + image.shape[0],
        pad_size : pad_size + image.shape[1],
    ] = image
    return padded


def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    image = pad_image(normalize_image(image))
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)

    rotate_and_scale = to_3x3(
        cv2.getRotationMatrix2D(center, float(rotation), float(scale))
    )

    transform = rotate_and_scale
    if flip_horizontal:
        transform = transform @ horizontal_flip_matrix(*center)

    transform = translation_matrix(translation_x, translation_y) @ transform

    return cv2.warpAffine(
        image,
        transform[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=WHITE,
    )


def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                scale = gr.Slider(
                    minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale"
                )
                rotation = gr.Slider(
                    minimum=-180,
                    maximum=180,
                    step=1,
                    value=0,
                    label="Rotation (degrees)",
                )
                translation_x = gr.Slider(
                    minimum=-300,
                    maximum=300,
                    step=10,
                    value=0,
                    label="Translation X",
                )
                translation_y = gr.Slider(
                    minimum=-300,
                    maximum=300,
                    step=10,
                    value=0,
                    label="Translation Y",
                )
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            image_output = gr.Image(label="Transformed Image")

        inputs = [
            image_input,
            scale,
            rotation,
            translation_x,
            translation_y,
            flip_horizontal,
        ]

        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo


if __name__ == "__main__":
    interactive_transform().launch()
