import cv2
import gradio as gr
import numpy as np


points_src = []
points_dst = []
image = None


def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = np.asarray(img).copy() if img is not None else None
    return image


def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    if image is None:
        return None

    x, y = int(evt.index[0]), int(evt.index[1])

    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 4, (0, 0, 255), -1)
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 4, (255, 0, 0), -1)

    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(
            marked_image,
            tuple(points_src[i]),
            tuple(points_dst[i]),
            (0, 255, 0),
            2,
            tipLength=0.25,
        )

    return marked_image


def normalize_control_points(source_pts, target_pts):
    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)

    if source_pts.size == 0 or target_pts.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    source_pts = source_pts.reshape(-1, 2)
    target_pts = target_pts.reshape(-1, 2)
    pair_count = min(len(source_pts), len(target_pts))
    return source_pts[:pair_count], target_pts[:pair_count]


def weighted_inverse_displacement(points, target_pts, source_pts, alpha, eps):
    displacements = source_pts - target_pts
    diff = points[:, None, :] - target_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.argmin(dist2, axis=1)
    exact = dist2[np.arange(len(points)), nearest] < eps

    weights = 1.0 / np.power(dist2 + eps, alpha)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    mapped = points + (weights @ displacements) / weights_sum

    if np.any(exact):
        mapped[exact] = source_pts[nearest[exact]]
    return mapped


def affine_mls_inverse_map(points, target_pts, source_pts, alpha, eps):
    diff = target_pts[None, :, :] - points[:, None, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.argmin(dist2, axis=1)
    exact = dist2[np.arange(len(points)), nearest] < eps

    weights = 1.0 / np.power(dist2 + eps, alpha)
    weights_sum = np.sum(weights, axis=1, keepdims=True)

    p_star = (weights @ target_pts) / weights_sum
    q_star = (weights @ source_pts) / weights_sum
    p_hat = target_pts[None, :, :] - p_star[:, None, :]
    q_hat = source_pts[None, :, :] - q_star[:, None, :]

    a = np.einsum("nk,nki,nkj->nij", weights, p_hat, p_hat)
    b = np.einsum("nk,nki,nkj->nij", weights, p_hat, q_hat)
    a = a + eps * np.eye(2, dtype=np.float64)[None, :, :]

    try:
        transform = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        transform = np.matmul(np.linalg.pinv(a), b)

    centered_points = points - p_star
    mapped = np.einsum("ni,nij->nj", centered_points, transform) + q_star

    if np.any(exact):
        mapped[exact] = source_pts[nearest[exact]]
    return mapped


def build_inverse_map(height, width, source_pts, target_pts, alpha, eps):
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    mapped = np.empty_like(points, dtype=np.float32)

    chunk_size = 65536
    for start in range(0, len(points), chunk_size):
        end = start + chunk_size
        chunk = points[start:end]
        if len(source_pts) < 3:
            mapped[start:end] = weighted_inverse_displacement(
                chunk, target_pts, source_pts, alpha, eps
            )
        else:
            mapped[start:end] = affine_mls_inverse_map(
                chunk, target_pts, source_pts, alpha, eps
            )

    map_x = mapped[:, 0].reshape(height, width)
    map_y = mapped[:, 1].reshape(height, width)
    return map_x, map_y


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    if image is None:
        return None

    image = np.asarray(image)
    source_pts, target_pts = normalize_control_points(source_pts, target_pts)
    if len(source_pts) == 0:
        return image.copy()

    height, width = image.shape[:2]
    map_x, map_y = build_inverse_map(
        height, width, source_pts, target_pts, float(alpha), float(eps)
    )

    if image.ndim == 2:
        border_value = 255
    else:
        border_value = tuple([255] * image.shape[2])

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def run_warping():
    global points_src, points_dst, image
    if image is None:
        return None

    return point_guided_deformation(
        image,
        np.asarray(points_src, dtype=np.float64),
        np.asarray(points_dst, dtype=np.float64),
    )


def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image


def build_demo():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    interactive=True,
                    width=800,
                    type="numpy",
                )
                point_select = gr.Image(
                    label="Click to Select Source and Target Points",
                    interactive=True,
                    width=800,
                    type="numpy",
                )

            with gr.Column():
                result_image = gr.Image(label="Warped Result", width=800, type="numpy")

        run_button = gr.Button("Run Warping")
        clear_button = gr.Button("Clear Points")

        input_image.upload(upload_image, input_image, point_select)
        input_image.change(upload_image, input_image, point_select)
        point_select.select(record_points, None, point_select)
        run_button.click(run_warping, None, result_image)
        clear_button.click(clear_points, None, point_select)

    return demo


if __name__ == "__main__":
    build_demo().launch()
