from data.wider_face import *
from models.model import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.decode import *
from config import *
import torch
from PIL import Image, ImageDraw

def detect_image(img_path):
    inp = cv2.imread(img_path)
    inp = cv2.resize(inp, (800, 800))

    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)
    input = torch.from_numpy(inp)

    heads = {
                'hm': 1,
                'wh': 2,
                "hm_offset": 2,
                "landmarks": 10
            }
    
    model = MobileNetSeg(base_name='mobilenetv2_10', heads=heads)

    weights = torch.load('saved/models/Center_Face/0711_103651/checkpoint-epoch29.pth')
    state_dict = weights['state_dict']

    load_model(model, state_dict)
    model.eval()

    with torch.no_grad():
        out = model(torch.unsqueeze(input, 0))

    out_hm = out[0]['hm']
    out_offset = out[0]['hm_offset']
    out_landmarks = out[0]['landmarks']
    out_wh = out[0]['wh']

    with torch.no_grad():
        decode = centerface_decode(out_hm, out_wh, out_landmarks, out_offset)

    img_decode = decode[0]
    bboxes = []
    for i in range(img_decode.shape[0]):
        bboxes.append(img_decode[i][:5])

    visualize_result(img_path, bboxes)


def visualize_result(img_path, anns, confidence = 0.3):
    img =  cv2.imread(img_path)
    img = img.transpose(1,2,0)
    fig, ax = plt.subplots()

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for ann in anns:
        # the segmentation are in format (x1, y1, x2, y2) (top left and bottom right points) 
        # (after convert from coco format to box format)
        # coco format (top left x, top left y, width, height)
        x1 = ann[0] 
        y1 = ann[1] 
        x2 = ann[2] 
        y2 = ann[3]

        if ann[4] > confidence:
            rect = patches.Rectangle((x1*4, y1*4), (x2-x1)*4, (y2-y1)*4, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    img_path = '../../data/wider_face/WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_859.jpg'
    detect_image(img_path)


