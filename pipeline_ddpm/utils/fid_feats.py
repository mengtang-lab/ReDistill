import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import torch

from inception import InceptionV3
from fid_score import get_activations


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
parser.add_argument("--img-size", type=int, default=32, help="Image size to set")
parser.add_argument(
    "--num-workers",
    type=int,
    help=(
        "Number of processes to use for data loading. " "Defaults to `min(8, num_cpus)`"
    ),
)
parser.add_argument(
    "--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu"
)
parser.add_argument(
    "--dims",
    type=int,
    default=2048,
    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
    help=(
        "Dimensionality of Inception features to use. "
        "By default, uses pool3 features"
    ),
)
parser.add_argument(
    "path",
    type=str,
    nargs=2,
    help=("Paths to .npy statistic files"),
)

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "JPEG", "pgm", "png", "ppm", "tif", "tiff", "webp"}



def compute_feats_of_path(path, model, batch_size, img_size, dims, device, num_workers=1):
    if path.endswith(".npz"):
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]
        assert False
    else:
        path = pathlib.Path(path)
        # files = sorted(
        #     [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        # )
        files = []
        for root, dirs, imgs in os.walk(path, topdown=True):
            files.extend([os.path.join(root, img) for img in imgs])
            # files.extend(img for ext in IMAGE_EXTENSIONS for img )
            # for name in imgs:
            #     print(os.path.join(root, name))
            # for name in dirs:
            #     print(os.path.join(root, name))
        files = sorted(files)
        feats = get_activations(files, model, batch_size, img_size, dims, device, num_workers)

    return feats

def save_fid_feats(paths, batch_size, img_size, device, dims, num_workers=1):
    """Saves FID statistics of one path"""
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError("Existing output file: %s" % paths[1])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    print(f"Saving statistics for {paths[0]}")

    feats = compute_feats_of_path(
        paths[0], model, batch_size, img_size, dims, device, num_workers
    )

    np.save(paths[1], feats)


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    save_fid_feats(args.path, args.batch_size, args.img_size, device, args.dims, num_workers)


if __name__ == "__main__":
    main()
