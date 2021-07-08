import os
import torch
from args import parser

# unnote if running locally
# DATA_DIR = '../../datasets/wider_face/'

# unnote if using colab
DATA_DIR = '/content/drive/Shareddrives/Giang/HaiNguyen/FaceDetection/dataset/wider_face/'

# unnote if using kaggle
# DATA_DIR = '../../input/widerface/wider_face/'

assert os.path.isdir(DATA_DIR)


ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations/')

assert os.path.isdir(ANNOTATIONS_DIR)
assert len(os.listdir(ANNOTATIONS_DIR)) > 0


"""
   Image directiries
"""
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'WIDER_train/images/')
assert os.path.isdir(TRAIN_IMAGES_DIR)

TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'WIDER_test/images/')
assert os.path.isdir(TEST_IMAGES_DIR)

VAL_IMAGES_DIR = os.path.join(DATA_DIR, 'WIDER_val/images/')
assert os.path.isdir(VAL_IMAGES_DIR)

"""
   Input and output resolution
"""
INPUT_SIZE = 800
OUTPUT_SIZE = 200

"""
   Data Augment
"""
# Choosing whether to use these technique or not
RANDOM_CROP = True
COLOR_AUG = True

SCALE = 0.4
SHIFT = 0.1

AUG_ROT = 0
ROTATE = 0

FLIP_PROB = 0.5

# Max objects to detect in an image
MAX_OBJETCS = 32

# Face class
NUM_CLASSES = 1

# 5 landmarks per face
NUM_LANMARKS = 5

DENSE_HP = False

DEBUG = 0

'''
Training parma
'''
SHUFFLE = True
BATCH_SIZE = 4

VALIDATION_SPLIT = 0.1
NUMBER_WORKER = 0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_STACKS = 1

EVAL_ORACLE_KPS_HM = False

EVAL_ORACLE_CENTER_HM = False

EVAL_ORACLE_KPS = False

EVAL_ORACLE_KPS_OFFSET = False

'''
Loss weight
'''
WH_WEIGHT = 0.1

REG_OFFSET = True

OFF_WEIGHT = 1

HM_WEIGHT = 1

LM_WEIGHT = 0.1
