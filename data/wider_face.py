import json

from torch.utils.data import Dataset
from data.utils import *
from config import *
from data.augment import *
from data.utils import *
from data.utils import *
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import re

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
# from torch._six import container_abcs, string_classes, int_classes
import collections
from torch._six import string_classes
from IPython import embed


np_str_obj_array_pattern = re.compile(r'[SaUO]')

# class WiderFaceDataset(FACEHP, MultiPoseDataset):
#     pass

class WiderFaceDataset(Dataset):
    def __init__(self, split):
        # super(WiderFaceDataset, self).__init__()

        self.split = split.lower()
        assert split in ['train', 'test', 'val']

        # Data directories
        self.data_dir = DATA_DIR
        self.ann_dir = ANNOTATIONS_DIR
        self.ann_paths = ''

        # Number of detect classes
        self.num_classes = 1 

        # Mean and var for standardzation
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.28863828, 0.27408164, 0.27809835],
                        dtype=np.float32).reshape(1, 1, 3)

        if self.split == 'train':
            self.img_dir = TRAIN_IMAGES_DIR
            self.ann_paths = os.path.join(ANNOTATIONS_DIR, 'train_wider_face.json')
        elif self.split == 'TEST':
            self.img_dir = TEST_IMAGES_DIR
        else:
            self.img_dir = VAL_IMAGES_DIR
            self.ann_paths = os.path.join(ANNOTATIONS_DIR, 'val_wider_face.json')

        # Max objects to be detected in an image
        self.max_objects = MAX_OBJETCS

        # Borrow from CenterFace.Pytorch repo
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                  [4, 6], [3, 5], [5, 6], 
                  [5, 7], [7, 9], [6, 8], [8, 10], 
                  [6, 12], [5, 11], [11, 12], 
                  [12, 14], [14, 16], [11, 13], [13, 15]]
    
        self.acc_idxs = [1, 2, 3, 4]

        self.flip_idx = [[0, 1], [3, 4]] 

        print('==> initializing centerface key point {} data.'.format(self.split))
        self.coco = coco.COCO(self.ann_paths)
        image_ids = self.coco.getImgIds()

        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
                else:
                    self.images = image_ids 
      
        # Number of samples in the dataset
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

        # Convert to float number
        def _to_float(self, x):
            return float("{:.2f}".format(x))

        # Convert to eval format
        def convert_eval_format(self, all_bboxes):
            detections = []
            for image_id in all_bboxes:
                for cls_ind in all_bboxes[image_id]:
                    category_id = 1
                    for dets in all_bboxes[image_id][cls_ind]:
                        bbox = dets[:4]
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = dets[4]
                        bbox_out  = list(map(self._to_float, bbox))
                        keypoints = np.concatenate([
                            np.array(dets[5:39], dtype=np.float32).reshape(-1, 2), 
                            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
                        keypoints  = list(map(self._to_float, keypoints))

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score)),
                            "keypoints": keypoints
                        }
                        detections.append(detection)
            return detections

    # Get len of the dataset
    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))


    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i


    def visualize_item(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        num_objs = len(anns)

        img = cv2.imread(img_path)
        
        fig, ax = plt.subplots()

        ax.imshow(img)

        for ann in anns:
            # the segmentation are in format (x1, y1, x2, y2) (top left and bottom right points) 
            # (after convert from coco format to box format)
            # coco format (top left x, top left y, width, height)
            x1 = ann['segmentation'][0]
            y1 = ann['segmentation'][1]
            x2 = ann['segmentation'][2]
            y2 = ann['segmentation'][3]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')

            ax.add_patch(rect)

            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        plt.show()

    def __getitem__(self, index):
        # Define basic variables
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = len(anns)

        input_res = INPUT_SIZE
        output_res = OUTPUT_SIZE
        num_joints = NUM_LANMARKS

        # If there are more than max number of objects, only detect max number of objects
        if num_objs > self.max_objects:
            num_objs = self.max_objects
            # if there are more than max objects, choose random max objects and save annotations
            anns = np.random.choice(anns, num_objs)

        # Reading image
        img = cv2.imread(img_path)

        # print(img_path)

        # Random scale image and annotation ?
        # img, anns = Data_anchor_sample(img, anns)
        # print(img.shape)
        # plt.imshow(img)

        # Define height, weight, center of the image
        height, width = img.shape[0], img.shape[1]

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        # Set flipped status
        flipped = False

        if self.split == 'train':
            if RANDOM_CROP:
                # s = s * np.random.choice(np.arange(0.8, 1.1, 0.1))
                s = s
                # _border = np.random.randint(128*0.4, 128*1.4)
                _border = s * np.random.choice([0.1, 0.2, 0.25])
                w_border = self._get_border(_border, img.shape[1])
                h_border = self._get_border(_border, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = SCALE
                cf = SHIFT
                c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < AUG_ROT:
                rf = ROTATE
                rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

            # Random flip with probability = FLIP
            if np.random.random() < FLIP_PROB:
                flipped = True
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1


            # Get transform matrix into input resolution (800x800) and apply to input image

        trans_input = get_affine_transform(c, s, rot, [input_res, input_res])
        inp = cv2.warpAffine(img, trans_input, (input_res, input_res), flags=cv2.INTER_LINEAR)

        # inp = cv2.resize(img, (input_res, input_res), interpolation = cv2.INTER_AREA)
        # inp = cv2.resize(img, (input_res, input_res))
        # print('after transform')
        # plt.imshow(inp)
        

        # Prerocessing input
        inp = (inp.astype(np.float32) / 255.)

        if self.split == 'train' and COLOR_AUG == 'True':                 # Random enhence iamge
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
            # inp = Randaugment(self._data_rng, inp, self._eig_val, self._eig_vec)
        
        # Standardize
        inp = (inp - self.mean) / self.std
        # Transpose shape into channel x width x height (or heigh x width)
        inp = inp.transpose(2, 0, 1)

        # Get transform matrix into output resolution (200x200) in 2 case: with rotation and without rotation
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])


        '''
        Init some ground truth mask, heatmap and some other labels: center, bbox shape and landmarks

        Center ground truth: 
            hm: 
                shape: num_classes x out_res x out_res (1 x 200 x 200)
                value: [0, 1]
                usage: creat heat map to detect whether a location is center or not (1 is center and 0 is no center)

            reg: (center_offset)
                shape: max_objects x 2 (32 x 2)
                value: R
                usage: calculate center offset (the deviation of center coordianate in output map) 
                    (x / R, y / R) -> (int(x / R), int(y / R)) 

            reg_mask: (center_offset_mask)
                shape: max_objects (32, )
                value: {0, 1} maybe ? 
                usage:  Whether it needs to be used to calculate the error  (1 is yes and 0 is no ?)

        hm_hp: (kps_hm)
            shape: num_joints x output_res x output_res (5 x 200 x 200)
            value: R
            usage: creat heat map to detect whether a location is keypoint or not 

        wh:
            shape: max_objects x 2 (32 x 2)
            value: R
            usage: creat a ground truth width and height of bounding box for each face in the image


        wight_mask: template unneccessary

        dense_kps: (temparutate not use)
            shape: num_joints x 2 x output_res x output_res (5 x 2 x 200 x 200) to draw dense reg ? 
            reshaped: (num_joints * 2) x output_res x output_res (10 x 200 x 200)
            value: R
            usage: 

        dense_kps_mask: (tempaturate not use)
            shape: num_joints x output_res x output_res
            value: {0, 1}
            usage: Whether it needs to be used to calculate the error  (1 is yes and 0 is no ?)

        kps:
            shape: max_objects x (num_joints * 2) (32 x 10)
            value: R
            usage: The deviation of the key point from the center of the face bbox for each object (5 joints per object)

        kps_mask:
            shape: max_object x (num_joints * 2) (32 x 10)
            value: {0, 1}
            usage: Whether it needs to be used to calculate the error  (1 is yes and 0 is no ?)

        hp_offset: (kps_offset)
            shape: (max_object * num_joints) x 2 (160 x 2)
            value: R
            usage: calculate keypoints offset (the deviation of keypoints coordianate in output map) 
                    (x / R, y / R) -> (int(x / R), int(y / R))

        hp_ind: (kps_ind)
            shape: (max_object * num_joints) (160, )
            value: R 
            usage: calculate index of 5 landmarks for each object 

        hp_mask: (kps_offset_mask)
            shape: (max_object * num_joints) (160, )
            value: {0, 1}
            usage: Calculate the mask of the loss (maybe 1 means use for calculate loss and 0 means no usage)
        '''

        # hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
        center_hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)

        # hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        kps_hm = np.zeros((num_joints, output_res, output_res), dtype=np.float32)

        dense_kps = np.zeros((num_joints, 2, output_res, output_res), dtype=np.float32)

        dense_kps_mask = np.zeros((num_joints, output_res, output_res), dtype=np.float32)

        wh = np.zeros((self.max_objects, 2), dtype=np.float32)

        kps = np.zeros((self.max_objects, num_joints * 2), dtype=np.float32)

        # reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        center_offset = np.zeros((self.max_objects, 2), dtype=np.float32)

        # reg_mask = np.zeros((self.max_objects), dtype=np.uint8)
        center_offset_mask = np.zeros((self.max_objects), dtype=np.uint8)

        center_ind = np.zeros((self.max_objects), dtype=np.int64)

        wight_mask = np.ones((self.max_objects), dtype=np.float32)

        kps_mask = np.zeros((self.max_objects, num_joints * 2), dtype=np.uint8)

        # hp_offset = np.zeros((self.max_objects * num_joints, 2), dtype=np.float32)
        kps_offset = np.zeros((self.max_objects * num_joints, 2), dtype=np.float32)

        # hp_ind = np.zeros((self.max_objects * num_joints), dtype=np.int64)
        kps_ind = np.zeros((self.max_objects * num_joints), dtype=np.int64)

        # hp_mask = np.zeros((self.max_objects * num_joints), dtype=np.int64)
        kps_offset_mask = np.zeros((self.max_objects * num_joints), dtype=np.int64)

        '''
        Define gaussian draw to draw
        '''
        # draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        draw_gaussian = draw_umich_gaussian

        '''
        Combine and define all type of Ground Truth needed for detecting task
        '''
        gt_det = []

        '''
        For each object in the image
        '''
        for k in range(num_objs):
            # ann for object k in coco format (a dictionary with keys are features and value are value of that features)
            ann = anns[k]
            # bbox of the object
            bbox = self._coco_box_to_bbox(ann['bbox'])
            # all object has category_id = 1 (face class)
            # define cls_id like this maybe made it easier in update GT value afterward
            cls_id = int(ann['category_id']) - 1

            # a numpy matrix with shape 5x3 for each object corresponding to 5 landmarks with x y coordinate and difficult ? 
            pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)

            # handing bounding box in case image is flipped
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

            # applying affine transform to bounding box to get bbox coordinate in the output heat map
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            # handling case that bounding box has coordinate that out of range with the output resolution
            bbox = np.clip(bbox, 0, output_res - 1)

            # height and weight of the bbox
            h_bbox, w_bbox = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if (h_bbox > 0 and w_bbox > 0) or (rot != 0):
                # round number upward to nearest integer
                radius = gaussian_radius((math.ceil(h_bbox), math.ceil(w_bbox)))

                # radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius)) 
                radius = max(0, int(radius))

                # Face center in the output map
                box_ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  

                # center point with int coordinate
                box_ct_int = box_ct.astype(np.int32)    

                '''
                Fill in pre-init label numpy arrrays
                '''

                # Center gaussian heatmap
                num_kpts = pts[:, 2].sum()                           # When there are no key points
                
                if num_kpts == 0:                                    # The samples without key points are more difficult samples
                    # print('No keypoints')
                    center_hm[cls_id, box_ct_int[1], box_ct_int[0]] = 0.9999

                # Index of face bbox in 1/4 feature map
                center_ind[k] = box_ct_int[1] * output_res + box_ct_int[0]
                
                # Center offset map 
                center_offset[k] = box_ct - box_ct_int 
                center_offset_mask[k] = 1
                
                # Width height map
                # The height and width of the face bbox, the way of the centerface paper
                wh[k] = np.log(1. * w_bbox / 4), np.log(1. * h_bbox / 4)  

                # Landmarks (Keypoints)
                hp_radius = gaussian_radius((math.ceil(h_bbox), math.ceil(w_bbox)))
                hp_radius = max(0, int(hp_radius))
                for j in range(num_joints):
                    if pts[j , 2] > 0:
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and pts[j, 1] >= 0 and pts[j, 1] < output_res:
                            # The deviation of the key point from the center of the face bbox
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - box_ct_int                
                            kps_mask[k, j * 2: j * 2 + 2] = 1

                            pt_int = pts[j, :2].astype(np.int32)
                            # Key point integerization deviation
                            kps_offset[k * num_joints + j] = pts[j, :2] - pt_int 
                            kps_offset_mask[k * num_joints + j] = 1
                            # Keypoints index in the 1D numpy array
                            kps_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]   

                            if DENSE_HP:
                                # must be before draw center hm gaussian
                                draw_dense_reg(dense_kps[j], center_hm[cls_id], box_ct_int, 
                                            pts[j, :2] - box_ct_int, radius, is_offset=True)
                                draw_gaussian(dense_kps_mask[j], box_ct_int, radius)

                            # Applying gaussian kernel for keypoints heatmap
                            draw_gaussian(kps_hm[j], pt_int, hp_radius)  

                            # If face are too small to detect, then skip
                            if ann['bbox'][2]*ann['bbox'][3] <= 16.0:
                                kps_mask[k, j * 2: j * 2 + 2] = 0
                
                # Applying gaussian kernel for center heatmap
                draw_gaussian(center_hm[cls_id], box_ct_int, radius)

                gt_det.append([box_ct[0] - w_bbox / 2, box_ct[1] - h_bbox / 2, 
                                box_ct[0] + w_bbox / 2, box_ct[1] + h_bbox / 2, 1] + 
                                pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

        # Handling rotation case
        # if rot != 0:
        #     center_hm = center_hm * 0 + 0.9999
        #     center_offset_mask *= 0
        #     kps_mask *= 0
        if rot != 0:
            center_hm = center_hm * 0 + 0.9999
            center_offset_mask *= 0
            kps_mask *= 0

        ret = {'input': inp, 'hm': center_hm, 'reg_mask': center_offset_mask, 'ind': center_ind, 'wh': wh,
                'landmarks': kps, 'hps_mask': kps_mask, 'wight_mask': wight_mask, 'hm_offset': center_offset,
                'hm_hp': kps_hm, 'hp_offset': kps_offset, 'hp_ind': kps_ind, 'hp_mask': kps_mask}

        if DENSE_HP:
            dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints * 2, output_res, output_res)
            ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
            del ret['hps'], ret['hps_mask']

        if DEBUG > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                    np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}

            ret['meta'] = meta
        
        return ret

_use_shared_memory = False

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


# Change default collate function

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def multipose_collate(batch):
  objects_dims = [d.shape[0] for d in batch]
  index = objects_dims.index(max(objects_dims))

  # one_dim = True if len(batch[0].shape) == 1 else False
  res = []
  for i in range(len(batch)):
      tres = np.zeros_like(batch[index], dtype=batch[index].dtype)
      tres[:batch[i].shape[0]] = batch[i]
      res.append(tres)

  return res


def Multiposebatch(batch):
  sample_batch = {}
  for key in batch[0]:
    if key in ['hm', 'input']:
      sample_batch[key] = default_collate([d[key] for d in batch])
    else:
      align_batch = multipose_collate([d[key] for d in batch])
      sample_batch[key] = default_collate(align_batch)

  return sample_batch


def _to_float(self, x):
    return float("{:.2f}".format(x))

def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
        for cls_ind in all_bboxes[image_id]:
            category_id = 1
            for dets in all_bboxes[image_id][cls_ind]:
                bbox = dets[:4]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = dets[4]
                bbox_out  = list(map(self._to_float, bbox))
                keypoints = np.concatenate([
                np.array(dets[5:39], dtype=np.float32).reshape(-1, 2), 
                np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
                keypoints  = list(map(self._to_float, keypoints))

                detection = {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": bbox_out,
                    "score": float("{:.2f}".format(score)),
                    "keypoints": keypoints
                }
                detections.append(detection)
    return detections

def __len__(self):
    return self.num_samples

def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))


def run_eval(self, results, save_dir):
# result_json = os.path.join(opt.save_dir, "results.json")
# detections  = convert_eval_format(all_boxes)
# json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()