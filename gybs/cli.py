"""Console script for GYBS_hackathon."""
import argparse
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import tifffile
from skimage.filters import threshold_minimum, threshold_triangle, threshold_mean
from skimage.morphology import disk, opening
from scipy import ndimage
import cv2


def main():
    """Console script for GYBS_hackathon."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',  # input .nii.gz image
        dest="in_file_path",
        type=str,
        required=True
    )
    parser.add_argument(
        '-o',  # output folder
        dest="out_path",
        type=str,
        required=False
    )
    parser.add_argument(
        '--orientation',
        dest="orientation",
        type=str,
        required=True
    )
    parser.add_argument(
        '-v',  # voxel spacing
        dest="resolution",
        nargs="+",
        required=True
    )
    args = parser.parse_args()

    out_folder = args.out_path
    if not out_folder:
        out_folder = os.path.join(os.getcwd(), 'registration_output')

    img = nib.load(args.in_file_path)  # Read nifti image
    img_np = np.array(img.dataobj)  # convert to numpy
    img_np = pre_process(img_np)  # pre-process image

    in_file_path = os.path.basename(args.in_file_path) + '.tif'
    tifffile.imwrite(in_file_path, img_np)

    # register image using brainreg (package developed by Brainglobe)
    launch_registration(in_file_path, out_folder, "allen_mouse_25um", args.resolution, args.orientation)


def pre_process(img):
    img_fft = np.zeros_like(img)
    for x in range(img.shape[-1]):
        if x % 10 == 0:
            print(f"{x} of {img.shape[-1]}")
        coronal_slice = img[:, :, x]
        foreground_mask = subtract_background_coronal_plane(coronal_slice)
        slice_fft = fft_2d_notch_filter(coronal_slice)
        slice_fft[foreground_mask == 0] = 0
        img_fft[:, :, x] = slice_fft
    return img_fft


def denoise_fft(image):
    """
    Apply circular mask to the image in FFT domain.
    """
    image_dtype = image.dtype
    img_float = image.astype(np.float32)

    H, W = img_float.shape
    img_fft = np.fft.fft2(img_float)/(W * H)

    img_fft = np.fft.fftshift(img_fft)

    center = [H//2, W//2]
    r = 200  # TODO hardcoded
    x, y = np.ogrid[:H, :W]
    mask_area = (x - center[0])**2 + (y - center[1])**2 >= r**2
    img_fft[mask_area] = 0

    img_fft = np.fft.ifftshift(img_fft)
    out_ifft = np.fft.ifft2(img_fft)

    image = (np.real(out_ifft) * W * H).astype(image_dtype)
    return image


def subtract_background_coronal_plane(img):
    """
    Subtract background from downsampled_standard.tif image, to compute overlap and nmi with atlas.

    Mask is computed from denoised image with low cutoff frequency
    (blured signigicantly) for better thresholding.
    """
    img_lf = denoise_fft(img)
    img_lf = img_lf.astype(np.float32)
    try:
        thr = threshold_triangle(img_lf)
    except:  # attempt to get argmax of an empty sequence
        return np.zeros_like(img).astype(np.uint16)

    mask = (img_lf > thr).astype(np.uint8)
    mask = (ndimage.binary_fill_holes(mask)).astype(np.uint8)
    kernel = disk(3)
    mask = (opening(mask, kernel)).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    nb_components -= 1
    for i in range(nb_components):
        if sizes[i] < 12000:  # TODO hardcoded
            mask[output == i + 1] = np.abs(mask[output == i + 1] - 1)

    return mask


def calculate_first_harmonic(image):
    """
    Calculate as argmax() of FFT computed on 2D image integrated over axis perpendicular to stripes.
    """
    img_sum = np.sum(image, axis=1)
    fft_seq = np.abs(np.fft.rfft(img_sum))/img_sum.shape[0]
    first_harmonic = np.argmax(fft_seq[10:]) + 10
    return first_harmonic


def ideal_notch_filter(fshift, points):
    d0 = 5.0  # cutoff frequency
    H, W = fshift.shape
    u, v = np.ogrid[:H, :W]
    for d in range(len(points)):
        u0, v0 = points[d]
        mask1 = (u - u0)**2 + (v - v0)**2 <= d0
        mask2 = (u + u0)**2 + (v + v0)**2 <= d0
        fshift[mask1] = 0
        fshift[mask2] = 0
    return fshift


def gaussian_notch_filter(fft_shift_img, points):
    d0 = 55  # cutoff frequency
    H, W = fft_shift_img.shape
    u, v = np.ogrid[:H, :W]
    for d in range(len(points)):
        u0, v0 = points[d]
        d1 = ((u - u0)**2 + (v - v0)**2)**0.5
        d2 = ((u + u0)**2 + (v + v0)**2)**0.5
        fft_shift_img[u,v] *= (1 - np.exp(-0.5 * (d1 * d2 / d0**2)))

    return fft_shift_img


def fft_2d_notch_filter(image, stripes_direction="h"):
    """
    Remove points corresponding to stripes in the 2D FFT plane using notch filter.

    stripes_direction: "h" (horizontal stripes) or "v" (vertical stripes)
    in fMOST data stripes are horizontal
    """
    image_dtype = image.dtype
    image = image.astype(np.float32)
    first_harmonic = calculate_first_harmonic(image)
    H, W = image.shape

    # do 2D fft
    img_fft = np.fft.fft2(image.astype(np.float32)) / (W * H)
    # shift fft to get maximum at the center
    img_fft = np.fft.fftshift(img_fft)

    # filter shifted fft
    points = []
    image_center = (H // 2, W // 2)
    print("Image center", image_center)
    if stripes_direction == "h":
        # compute points on vertical axis of symmetry
        for point_ind in range(1, image_center[0] // first_harmonic):
            points.extend([
                [image_center[0] + point_ind * first_harmonic, image_center[1]],
                [image_center[0] - point_ind * first_harmonic, image_center[1]]
            ])
    elif stripes_direction == "v":
        # compute points on horizontal axis of symmetry
        for point_ind in range(1, image_center[1] // first_harmonic):
            points.extend([
                [image_center[0], image_center[1] + point_ind * first_harmonic],
                [image_center[0], image_center[1] - point_ind * first_harmonic]
            ])
    else:
        raise NotImplemented("Can only automatically remove vertical or horizontal stripes")
    points = np.asarray(points)

    filtered_fft_shift = ideal_notch_filter(img_fft, points)
    # filtered_fft_shift = gaussian_notch_filter(img_fft, points)  # slower but slightly better quality

    # unshift
    img_fft = np.fft.ifftshift(filtered_fft_shift)
    # do inverse fft
    out_ifft = np.fft.ifft2(img_fft)
    image = (np.abs(out_ifft) * W * H).astype(image_dtype)
    return image


def launch_registration(input_file, output_folder, atlas, resolution, orientation):
    cmd = [
        'brainreg', input_file, output_folder,
        '-v', resolution[0], resolution[1], resolution[2],
        '--orientation', orientation,
        '--atlas', atlas,
        '--debug'
    ]

    ret = subprocess.run(cmd)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
