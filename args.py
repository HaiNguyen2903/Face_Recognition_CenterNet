import argparse
import os

from numpy import int32

parser = argparse.ArgumentParser(description = 'Arguments for configuring training process')

# data root dir
parser.add_argument('--data_dir', default = '../../datasets/wider_face/', type=str)

# input image resolution
parser.add_argument('--input_res', default = 800, type = int32)
parser.add_argument('--output_res', default = 200, type = int32)

# training params
parser.add_argument('--batch_size', default = 16, type = int32)
parser.add_argument('-val_split', default = 0.1)
parser.add_argument('--num_workers', default = 2, type = int32)



