import cv2
import numpy as np
import random
import math
import torch
import torchvision.transforms.functional as F


# def _get_param_spatial_crop(
#     scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
# ):
#     """
#     Given scale, ratio, height and width, return sampled coordinates of the videos.
#     """
#     for _ in range(num_repeat):
#         area = height * width
#         target_area = random.uniform(*scale) * area
#         if log_scale:
#             log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
#             aspect_ratio = math.exp(random.uniform(*log_ratio))
#         else:
#             aspect_ratio = random.uniform(*ratio)

#         w = int(round(math.sqrt(target_area * aspect_ratio)))
#         h = int(round(math.sqrt(target_area / aspect_ratio)))

#         if np.random.uniform() < 0.5 and switch_hw:
#             w, h = h, w

#         if 0 < w <= width and 0 < h <= height:
#             i = random.randint(0, height - h)
#             j = random.randint(0, width - w)
#             return i, j, h, w

#     # Fallback to central crop
#     in_ratio = float(width) / float(height)
#     if in_ratio < min(ratio):
#         w = width
#         h = int(round(w / min(ratio)))
#     elif in_ratio > max(ratio):
#         h = height
#         w = int(round(h * max(ratio)))
#     else:  # whole image
#         w = width
#         h = height
#     i = (height - h) // 2
#     j = (width - w) // 2
#     return i, j, h, w

# def random_sized_crop_img(
#     im,
#     size,
#     jitter_scale=(0.08, 1.0),
#     jitter_aspect=(3.0 / 4.0, 4.0 / 3.0),
#     max_iter=10,
# ):
#     """
#     Performs Inception-style cropping (used for training).
#     """
#     assert (
#         len(im.shape) == 3
#     ), "Currently only support image for random_sized_crop"
#     h, w = im.shape[1:3]
#     i, j, h, w = _get_param_spatial_crop(
#         scale=jitter_scale,
#         ratio=jitter_aspect,
#         height=h,
#         width=w,
#         num_repeat=max_iter,
#         log_scale=False,
#         switch_hw=True,
#     )
#     cropped = im[:, i : i + h, j : j + w]
#     return torch.nn.functional.interpolate(
#         cropped.unsqueeze(0),
#         size=(size, size),
#         mode="bilinear",
#         align_corners=False,
#     ).squeeze(0)


# def random_short_side_scale_jitter(
#     images, min_size, max_size, boxes=None, inverse_uniform_sampling=False
# ):
#     """
#     Perform a spatial short scale jittering on the given images and
#     corresponding boxes.
#     Args:
#         images (tensor): images to perform scale jitter. Dimension is
#             `num frames` x `channel` x `height` x `width`.
#         min_size (int): the minimal size to scale the frames.
#         max_size (int): the maximal size to scale the frames.
#         boxes (ndarray): optional. Corresponding boxes to images.
#             Dimension is `num boxes` x 4.
#         inverse_uniform_sampling (bool): if True, sample uniformly in
#             [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
#             scale. If False, take a uniform sample from [min_scale, max_scale].
#     Returns:
#         (tensor): the scaled images with dimension of
#             `num frames` x `channel` x `new height` x `new width`.
#         (ndarray or None): the scaled boxes with dimension of
#             `num boxes` x 4.
#     """
#     if inverse_uniform_sampling:
#         size = int(
#             round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
#         )
#     else:
#         size = int(round(np.random.uniform(min_size, max_size)))

#     height = images.shape[2]
#     width = images.shape[3]
#     if (width <= height and width == size) or (
#         height <= width and height == size
#     ):
#         return images, boxes
#     new_width = size
#     new_height = size
#     if width < height:
#         new_height = int(math.floor((float(height) / width) * size))
#         if boxes is not None:
#             boxes = boxes * float(new_height) / height
#     else:
#         new_width = int(math.floor((float(width) / height) * size))
#         if boxes is not None:
#             boxes = boxes * float(new_width) / width

#     return (
#         torch.nn.functional.interpolate(
#             images,
#             size=(new_height, new_width),
#             mode="bilinear",
#             align_corners=False,
#         ),
#         boxes,
#     )


def adjust_short(image, target_size=(224, 224), short_side_range=(256, 320)):
    # 获取原始图像的尺寸
    # image_height, image_width, _ = image.shape
    _, image_height, image_width = image.shape  # 获取原始尺寸
    
    # (1) 随机调整短边的大小
    short_side = random.randint(short_side_range[0], short_side_range[1])  # 在范围内随机选择短边的长度
    
    # 确定长边的缩放比例
    if image_height < image_width:
        # 高度为短边，按比例缩放宽度
        new_height = short_side
        new_width = int(image_width * (short_side / image_height))
    else:
        # 宽度为短边，按比例缩放高度
        new_width = short_side
        new_height = int(image_height * (short_side / image_width))
    
    # 缩放图像到新的尺寸
    # resized_image = cv2.resize(image, (new_width, new_height))
    resized_image = F.resize(image, (new_height, new_width))  
    
    # # (2) 居中裁剪为正方形
    # crop_size = min(new_height, new_width)
    
    # # 计算裁剪区域的起始位置（居中裁剪）
    # x_offset = (new_width - crop_size) // 2
    # y_offset = (new_height - crop_size) // 2
    
    # cropped_image = resized_image[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
    
    # # (3) 调整裁剪后的图像为224x224
    # final_image = cv2.resize(cropped_image, target_size)
    # (3) 居中裁剪为正方形
    crop_size = min(new_height, new_width)
    cropped_image = F.center_crop(resized_image, crop_size)  # 使用 center_crop 居中裁剪

    # (4) 调整到最终 target_size
    final_image = F.resize(cropped_image, target_size)  # 最终缩放到目标尺寸
    
    return final_image


def random_crop(image):
    """
    随机裁剪图像，裁剪区域占原图 8%-100%，长宽比随机选取 [0.25, 0.75] 之间。
    适用于 `torch.Tensor` 格式的图像，输入形状为 (C, H, W)。
    
    :param image: 输入的图像 (C, H, W) 格式，类型为 torch.Tensor
    :return: 裁剪后的图像，仍然是 torch.Tensor
    """
    # (1) 随机采样目标像素数目（8%-100%）
    # image_height, image_width, _ = image.shape
    _, image_height, image_width = image.shape  # 获取原始尺寸
    area = image_height * image_width
    crop_percentage = random.uniform(0.08, 1.0)  # 随机选取8%-100%之间的百分比
    target_area = int(area * crop_percentage)
    
    # (2) 在[1/4, 3/4]范围内随机采样一个长宽比
    aspect_ratio = random.uniform(0.25, 0.75)  # 随机选取长宽比

    # 计算裁剪的长和宽
    crop_width = int(np.sqrt(target_area * aspect_ratio))
    crop_height = target_area // crop_width

    # 确保裁剪区域不超出图片尺寸
    crop_width = min(crop_width, image_width)
    crop_height = min(crop_height, image_height)

    # # (3) 随机裁剪区域
    # x_offset = random.randint(0, image_width - crop_width)
    # y_offset = random.randint(0, image_height - crop_height)

    # cropped_image = image[y_offset:y_offset+crop_height, x_offset:x_offset+crop_width]

     # (3) 随机确定裁剪的起始坐标
    x_offset = random.randint(0, image_width - crop_width) if image_width > crop_width else 0
    y_offset = random.randint(0, image_height - crop_height) if image_height > crop_height else 0

    # 进行随机裁剪
    cropped_image = F.crop(image, y_offset, x_offset, crop_height, crop_width)

    return cropped_image

# 示例使用
if __name__ == "__main__":
    # 加载图像
    image = cv2.imread("example.jpg")

    # 调用函数进行随机裁剪和调整大小
    output_image = random(image)

    # 显示裁剪后的图像
    cv2.imshow("Random Crop and Resize", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
