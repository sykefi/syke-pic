"""Image related helper functions and classes."""

import random

import cv2
import numpy as np


class Compose:
    """PyTorch inspired composite class with more functionality.

    Used to join tranformation classes (data augmentations)
    into one callable object.
    """

    def __init__(self, transforms, target_dims, border):
        self.transforms = transforms
        self.target_dims = target_dims
        self.border = border
        if self.border == 'white':
            self.border = (255, 255, 255)
        elif self.border == 'black':
            self.border = (0, 0, 0)

    def __call__(self, img):
        if self.border == 'mode':
            mode = mode_pixel_value(img)
            border = (mode, mode, mode)
        else:
            border = self.border
        # Original image shape is needed for translation
        h, w = img.shape[:2]
        target_h, target_w = self.target_dims
        new_h, new_w = get_new_dims(h, w, target_h, target_w)

        for t in self.transforms:
            if isinstance(t, Resize):
                img = t(img, (new_h, new_w), self.target_dims, border)
            elif isinstance(t, Translate):
                # Translate original image's shorter side
                # i.e. the one with padding.
                # Calculate limit based on original and target image size.
                height = False
                width = False
                if h > w:
                    width = True
                    limit = int((target_w - new_w)/2.5)
                elif w > h:
                    height = True
                    limit = int((target_h - new_h)/2.5)
                img = t(img, limit, border, height, width)
            elif isinstance(t, (Zoom, Rotate)):
                img = t(img, border)
            else:
                img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize:
    def __call__(self, img, new_dims, target_dims, border=None):
        if border:
            img = resize_with_border(img, new_dims, target_dims, border)
        else:
            new_h, new_w = new_dims
            img = cv2.resize(img, (new_w, new_h))
        return img

    def __repr__(self):
        return 'Resize()'


class FlipHorizontal:
    def __call__(self, img):
        apply_flip = random.getrandbits(1)
        if apply_flip:
            img = cv2.flip(img, 1)
        return img

    def __repr__(self):
        return 'FlipHorizontal()'


class FlipVertical:
    def __call__(self, img):
        apply_flip = random.getrandbits(1)
        if apply_flip:
            img = cv2.flip(img, 0)
        return img

    def __repr__(self):
        return 'FlipVertical()'


class Translate:
    def __call__(self, img, limit, border=None, height=True, width=True):
        x = 0
        y = 0
        if height:
            y = random.randint(-limit, limit)
        if width:
            x = random.randint(-limit, limit)
        M = np.float32([[1, 0, x], [0, 1, y]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderValue=border)
        return img

    def __repr__(self):
        return 'Translate()'


class Zoom:
    """Assumes image is square i.e. already resized"""

    def __init__(self, zoom_range):
        self.zoom_range = zoom_range

    def __call__(self, img, border):
        fx = fy = round(random.uniform(*self.zoom_range), 2)
        h, w = img.shape[:2]
        img = cv2.resize(img, None, fx=fx, fy=fy,
                         interpolation=cv2.INTER_LINEAR)
        zh, zw = img.shape[:2]
        if fx < 1:
            pad_1 = int((w - zw) / 2)
            pad_2 = w - zw - pad_1
            img = cv2.copyMakeBorder(img, pad_1, pad_2, pad_1, pad_2,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=border)
        else:
            crop_1 = (zw - w) / 2
            crop_2 = zw - crop_1
            crop_1 = int(crop_1)
            crop_2 = int(crop_2)
            img = img[crop_1:crop_2, crop_1:crop_2]
        return img

    def __repr__(self):
        return f'Zoom(range={self.zoom_range})'


class Rotate:
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, img, border):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        angle = random.randint(-self.max_angle, self.max_angle)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=border)
        return img

    def __repr__(self):
        return f'Rotate(max_angle={self.max_angle})'


class ChangeBrightness:
    def __init__(self, brightness_range):
        self.brightness_range = brightness_range

    def __call__(self, img):
        value = random.uniform(*self.brightness_range)
        img = img*value
        img = img.clip(0, 255).astype(np.uint8)
        return img

    def __repr__(self):
        return f'ChangeBrightness(brightness_range={self.brightness_range})'


def get_new_dims(h, w, target_h, target_w):
    """Returns new dimensions that respect the old aspect ratio.

    `target_h` and `target_w` are usually the same value, since
    target dimensions are most likely squares.
    """

    if h > w:
        r = target_h / float(h)
        new_h = target_h
        new_w = int(w * r)
    else:
        r = target_w / float(w)
        new_h = int(h * r)
        new_w = target_w
    return new_h, new_w


def resize_with_border(img, new_dims, target_dims, border=(0, 0, 0),
                       interpolation=cv2.INTER_LINEAR):
    """Resize image while maintaining aspect ratio."""

    new_h, new_w = new_dims
    target_h, target_w = target_dims
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    h, w = img.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    b, g, r = border
    img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right,
                             borderType=cv2.BORDER_CONSTANT, value=[b, g, r])
    return img


def mode_pixel_value(img):
    """Returns the most common pixel value for image i.e. mode value."""

    # OpenCV's calcHist() is supposedly 40X faster than np.histogram().
    # There seems to be a difference whether image is passed as a list or
    # not to calcHist, even though both work. List seems to be the correct way.
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    mode = int(np.argmax(hist))
    return mode


def calculate_mean_std(img_paths, grayscale=False):
    """Calculate the mean and standard deviation images.

    Parameters
    ----------
    img_paths : iterable
        List of image paths that will be used in calculation.
    grayscale : bool
        Read images as grayscale, which will affect return value.

    Returns
    -------
    numpy.Array, numpy.Array
        Two lists, each with values for every dimension
        (3 by default, but 1 with `grayscale` set to True).
    """

    mean_sum = .0
    std_sum = .0
    img_paths = list(img_paths)
    for path in img_paths:
        if grayscale:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(path))
        mean, std = cv2.meanStdDev(img)
        mean_sum += mean
        std_sum += std
    mean = mean_sum / len(img_paths) / 255.0
    mean = np.squeeze(mean, axis=1)
    std = std_sum / len(img_paths) / 255.0
    std = np.squeeze(std, axis=1)
    return mean, std


def calculate_mean_dims(img_paths):
    """Calculate the mean image dimension for images.

    Parameters
    ----------
    img_paths : iterable
        List of image paths that will be used in calculation.

    Returns
    -------
    int, int
        Rounded mean values for height and width respectively.
    """

    height = 0.
    width = 0.
    for i, path in enumerate(img_paths, start=1):
        img = cv2.imread(str(path))
        h, w, c = img.shape
        height += h
        width += w
    height /= i
    width /= i
    return int(height), int(width)
