#!/usr/bin/env python

import numpy as np
import argparse
import time
from multiprocessing import Pool
from julia_curve import c_from_group

# Update according to your group size and number (see TUWEL)
GROUP_SIZE = 3
GROUP_NUMBER = 13

# Do not modify BENCHMARK_C
BENCHMARK_C = complex(-0.2, -0.65)

def compute_patch(args):
    x, y, patch_x, patch_y, size, xmin, xmax, ymin, ymax, c, max_iter, threshold = args
    img_patch = np.zeros((patch_x, patch_y), dtype=np.float32)
    xwidth = xmax - xmin
    yheight = ymax - ymin
    for i in range(patch_x):
        for j in range(patch_y):
            zx = xmin + (x + i) * xwidth / size
            zy = ymin + (y + j) * yheight / size
            z = complex(zx, zy)
            nit = 0
            while abs(z) <= threshold and nit < max_iter:
                z = z**2 + c
                nit += 1
            ratio = nit / max_iter
            img_patch[i, j] = ratio
    return (x, y, img_patch)

def compute_julia_in_parallel(size, xmin, xmax, ymin, ymax, patch, nprocs, c):
    max_iter = 300
    threshold = 10
    task_list = []
    for x in range(0, size, patch):
        for y in range(0, size, patch):
            actual_patch_x = min(patch, size - x)
            actual_patch_y = min(patch, size - y)
            task_list.append((x, y, actual_patch_x, actual_patch_y, size, xmin, xmax, ymin, ymax, c, max_iter, threshold))

    pool = Pool(nprocs)
    completed_patches = pool.map(compute_patch, task_list, chunksize=1)

    julia_img = np.zeros((size, size), dtype=np.float32)
    for x, y, img_patch in completed_patches:
        julia_img[x:x+img_patch.shape[0], y:y+img_patch.shape[1]] = img_patch

    return julia_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="image and height in pixels", type=int, default=500)
    parser.add_argument("--xmin", help="", type=float, default=-1.5)
    parser.add_argument("--xmax", help="", type=float, default=1.5)
    parser.add_argument("--ymin", help="", type=float, default=-1.5)
    parser.add_argument("--ymax", help="", type=float, default=1.5)
    parser.add_argument("--group-size", help="", type=int, default=None)
    parser.add_argument("--group-number", help="", type=int, default=None)
    parser.add_argument("--patch", help="patch width in pixels", type=int, default=20)
    parser.add_argument("--nprocs", help="number of workers", type=int, default=1)
    parser.add_argument("--draw-axes", help="Whether to draw axes", action="store_true")
    parser.add_argument("-o", help="output file")
    parser.add_argument("--benchmark", help="Whether to execute the script with the benchmark Julia set", action="store_true")
    args = parser.parse_args()

    if args.group_size is not None:
        GROUP_SIZE = args.group_size
    if args.group_number is not None:
        GROUP_NUMBER = args.group_number

    # Assign c based on mode
    c = BENCHMARK_C if args.benchmark else c_from_group(GROUP_SIZE, GROUP_NUMBER)

    stime = time.perf_counter()
    julia_img = compute_julia_in_parallel(
        args.size,
        args.xmin, args.xmax,
        args.ymin, args.ymax,
        args.patch,
        args.nprocs,
        c)
    rtime = time.perf_counter() - stime

    print(f"{args.size};{args.patch};{args.nprocs};{rtime}")

    if args.o:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(julia_img, interpolation='nearest', cmap=plt.get_cmap("hot"))

        if args.draw_axes:
            im_width = args.size
            im_height = args.size
            xmin = args.xmin
            xmax = args.xmax
            xwidth = args.xmax - args.xmin
            ymin = args.ymin
            ymax = args.ymax
            yheight = args.ymax - args.ymin

            xtick_labels = np.linspace(xmin, xmax, 7)
            ax.set_xticks([(x-xmin) / xwidth * im_width for x in xtick_labels])
            ax.set_xticklabels(['{:.1f}'.format(xtick) for xtick in xtick_labels])
            ytick_labels = np.linspace(ymin, ymax, 7)
            ax.set_yticks([(y-ymin) / yheight * im_height for y in ytick_labels])
            ax.set_yticklabels(['{:.1f}'.format(-ytick) for ytick in ytick_labels])
            ax.set_xlabel("Imag")
            ax.set_ylabel("Real")
        else:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(args.o, bbox_inches='tight')
