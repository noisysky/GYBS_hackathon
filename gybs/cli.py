"""Console script for GYBS_hackathon."""
import argparse
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import tifffile


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
        out_folder = 'registration_output'

    img = nib.load(args.in_file_path)  # Read nifti image
    img_np = np.array(img.dataobj)  # convert to numpy
    img_np = pre_process(img_np)  # pre-process image

    in_file_path = os.path.basename(args.in_file_path) + '.tif'
    tifffile.imwrite(in_file_path, img_np)

    # register image using brainreg (package developed by Brainglobe)
    launch_registration(in_file_path, out_folder, "allen_mouse_25um", args.resolution, args.orientation)


def pre_process(img):
    return img


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
