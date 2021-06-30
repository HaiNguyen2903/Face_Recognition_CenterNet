import torch.nn.functional as F
from models.multi_pose_loss import MultiPoseLoss
import pickle

opt = None

multi_pose_loss = MultiPoseLoss()


def center_face_loss(output, target):
    return multi_pose_loss.forward(output, target)


def nll_loss(output, target):
    return F.nll_loss(output, target)
