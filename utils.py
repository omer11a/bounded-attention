import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from sklearn.decomposition import PCA
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

import os


def display_attention_maps(
    attention_maps,
    is_cross,
    num_heads,
    tokenizer,
    prompts,
    dir_name,
    step,
    layer,
    resolution,
    is_query=False,
    is_key=False,
    points=None,
    image_path=None,
):
    attention_maps = attention_maps.reshape(-1, num_heads, attention_maps.size(-2), attention_maps.size(-1))
    num_samples = len(attention_maps) // 2
    attention_type = 'cross' if is_cross else 'self'
    for i, attention_map in enumerate(attention_maps):
        if is_query:
            attention_type = f'{attention_type}_queries'
        elif is_key:
            attention_type = f'{attention_type}_keys'

        cond = 'uncond' if i < num_samples else 'cond'
        i = i % num_samples
        cur_dir_name = f'{dir_name}/{resolution}/{attention_type}/{layer}/{cond}/{i}'
        os.makedirs(cur_dir_name, exist_ok=True)

        if is_cross and not is_query:
            fig = show_cross_attention(attention_map, tokenizer, prompts[i % num_samples])
        else:
            fig = show_self_attention(attention_map)
            if points is not None:
                point_dir_name = f'{cur_dir_name}/points'
                os.makedirs(point_dir_name, exist_ok=True)
                for j, point in enumerate(points):
                    specific_point_dir_name = f'{point_dir_name}/{j}'
                    os.makedirs(specific_point_dir_name, exist_ok=True)
                    point_path = f'{specific_point_dir_name}/{step}.png'
                    point_fig = show_individual_self_attention(attention_map, point, image_path=image_path)
                    point_fig.save(point_path)
                    point_fig.close()

        fig.save(f'{cur_dir_name}/{step}.png')
        fig.close()


def text_under_image(image: np.ndarray, text: str, text_color: tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    lines = text.splitlines()
    img = np.ones((h + offset + (text_size[1] + 2) * len(lines) - 2, w, c), dtype=np.uint8) * 255
    img[:h, :w] = image

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, 1, 2)[0]
        text_x, text_y = ((w - text_size[0]) // 2, h + offset + i * (text_size[1] + 2))
        cv2.putText(img, line, (text_x, text_y), font, 1, text_color, 2)

    return img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    return Image.fromarray(image_)


def show_cross_attention(attention_maps, tokenizer, prompt, k_norms=None, v_norms=None):
    attention_maps = attention_maps.mean(dim=0)
    res = int(attention_maps.size(-2) ** 0.5)
    attention_maps = attention_maps.reshape(res, res, -1)
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    if k_norms is not None:
        k_norms = k_norms.round(decimals=1)
    if v_norms is not None:
        v_norms = v_norms.round(decimals=1)
    images = []
    for i in range(len(tokens) + 5):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.detach().cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        token = tokens[i] if i < len(tokens) else tokens[-1]
        text = decoder(int(token))
        if k_norms is not None and v_norms is not None:
            text += f'\n{k_norms[i]}\n{v_norms[i]})'
        image = text_under_image(image, text)
        images.append(image)
    return view_images(np.stack(images, axis=0))


def show_queries_keys(queries, keys, colors, labels):  # [h ni d]
    num_queries = [query.size(1) for query in queries]
    num_keys = [key.size(1) for key in keys]
    h, _, d = queries[0].shape

    data = torch.cat((*queries, *keys), dim=1)  # h n d
    data = data.permute(1, 0, 2)  # n h d
    data = data.reshape(-1, h * d).detach().cpu().numpy()
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)  # n 2

    query_indices = np.array(num_queries).cumsum()
    total_num_queries = query_indices[-1]
    queries = np.split(data[:total_num_queries], query_indices[:-1])
    if len(num_keys) == 0:
        keys = [None, ] * len(labels)
    else:
        key_indices = np.array(num_keys).cumsum()
        keys = np.split(data[total_num_queries:], key_indices[:-1])

    fig, ax = plt.subplots()
    marker_size = plt.rcParams['lines.markersize'] ** 2
    query_size = int(1.25 * marker_size)
    key_size = int(2 * marker_size)
    for query, key, color, label in zip(queries, keys, colors, labels):
        print(f'# queries of {label}', query.shape[0])
        ax.scatter(query[:, 0], query[:, 1], s=query_size, color=color, marker='o', label=f'"{label}" queries')

        if key is None:
            continue

        print(f'# keys of {label}', key.shape[0])
        keys_label = f'"{label}" key'
        if key.shape[0] > 1:
            keys_label += 's'
        ax.scatter(key[:, 0], key[:, 1], s=key_size, color=color, marker='x', label=keys_label)

    ax.set_axis_off()
    #ax.set_xlabel('X-axis')
    #ax.set_ylabel('Y-axis')
    #ax.set_title('Scatter Plot with Circles and Crosses')

    #ax.legend()
    return fig


def show_self_attention(attention_maps):  # h n m
    attention_maps = attention_maps.transpose(0, 1).flatten(start_dim=1).detach().cpu().numpy()
    pca = PCA(n_components=3)
    pca_img = pca.fit_transform(attention_maps)  # N X 3
    h = w = int(pca_img.shape[0] ** 0.5)
    pca_img = pca_img.reshape(h, w, 3)
    pca_img_min = pca_img.min(axis=(0, 1))
    pca_img_max = pca_img.max(axis=(0, 1))
    pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
    pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    pca_img = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(pca_img)
    return pca_img


def draw_box(pil_img, bboxes, colors=None, width=5):
    draw = ImageDraw.Draw(pil_img)
    #font = ImageFont.truetype('./FreeMono.ttf', 25)
    w, h = pil_img.size
    colors = ['red'] * len(bboxes) if colors is None else colors
    for obj_bbox, color in zip(bboxes, colors):
        x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
        draw.rectangle([int(x_0 * w), int(y_0 * h), int(x_1 * w), int(y_1 * h)], outline=color, width=width)
    return pil_img


def show_individual_self_attention(attn, point, image_path=None):
    resolution = int(attn.size(-1) ** 0.5)
    attn = attn.mean(dim=0).reshape(resolution, resolution, resolution, resolution)
    attn = attn[round(point[1] * resolution), round(point[0] * resolution)]
    attn = (attn - attn.min()) / (attn.max() - attn.min())
    image = None if image_path is None else Image.open(image_path).convert('RGB')
    image = show_image_relevance(attn, image=image)
    return Image.fromarray(image)


def show_image_relevance(image_relevance, image: Image.Image = None, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        img = img.resize((relevnace_res ** 2, relevnace_res ** 2))
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    vis = image_relevance if image is None else show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis
