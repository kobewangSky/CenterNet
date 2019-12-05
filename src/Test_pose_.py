import sys
CENTERNET_PATH = './lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts
import argparse
import glob
import os
import cv2
import json
import numpy as np
from scipy.spatial import distance

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


actual = [1,1,1,0,0,0]
predictions = [0.9,0.9,0.9,0.1,0.1,0.1]


model_load_path = 'privacy_victor_v3_real_Virtual_victor_v1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', dest='model_path', default='../exp/multi_pose/{}//model_last.pth'.format(model_load_path), help='load model pth')
# parser.add_argument('--model_path', dest='model_path', default='../models/multi_pose_hg_3x.pth', help='load model pth')
# parser.add_argument('--data_path', dest='data_path', default='/media/ssd_external/2018-12-05-subset/images', help='load data dir')
# parser.add_argument('--label_path', dest='label_path', default='/media/ssd_external/2018-12-05-subset/annotations/output.json', help='load data dir')
parser.add_argument('--threshold', dest='threshold', default=0.2, help='pck threshold')

# parser.add_argument('--model_path', dest='model_path', default='../exp/multi_pose/hg_virtual/model_best.pth', help='load model pth')
parser.add_argument('--data_path', dest='data_path', default='../data/victor_v3/images_privacy/', help='load data dir')
parser.add_argument('--label_path', dest='label_path', default='../data/victor_v3/annotations/output.json', help='load data dir')


parser.add_argument('--visualize', dest='visualize', default=False, help='load data dir')
args = parser.parse_args()
MODEL_PATH = args.model_path
TASK = 'multi_pose'
opt = opts().init('{} --load_model {} --arch hourglass --nms'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
# img_list = sorted(glob.glob(os.path.join(DATA_PATH, '*.jpg')))
color = [(255, 0, 0), (0, 255, 0), (0, 255, 255)]
labels = json.load(open(args.label_path))
gts = {}
for anno in labels['annotations']:
    img_id = anno['image_id']

    #for merge eye ear to nose
    for it in range(1, 5, 1):
        anno['keypoints'][it * 3 ] = anno['keypoints'][0]
        anno['keypoints'][it * 3 +1] = anno['keypoints'][1]


    if img_id in list(gts.keys()):
        gts[img_id].append({'bbox': anno['bbox'],
                            'keypoints': anno['keypoints']})
    else:
        gts[img_id] = [{'bbox': anno['bbox'],
                            'keypoints': anno['keypoints']}]

img_list = labels['images']
dist_keypoints = []
index = 0


def getacc(scc_list, name):
    pck_correct = [0 for _ in range(17)]
    num_samples = [len(scc_list) for _ in range(17)]
    for dist_keypoint in scc_list:
        for idx, dist in enumerate(dist_keypoint):
            if dist < 0:
                num_samples[idx] -= 1
            elif dist <= args.threshold:
                pck_correct[idx] += 1
    pck_correct = [p / n for p, n in zip(pck_correct, num_samples)]
    print(name + 'PCK all: {}'.format(sum(pck_correct) / len(pck_correct)))

All = [i for i in range(138)]
stand = np.concatenate((All[0:25], All[77:84], All[133:138]))
sit = np.concatenate((All[25:39], All[46:60], All[72:77], All[84:96]))
lay = np.concatenate((All[39:46], All[60:72], All[96:133]))

stand_acc = []
sit_acc = []
lay_acc = []
temp_acc = []
for img in img_list:
    if index in stand:
        temp_acc = stand_acc
    elif index in sit:
        temp_acc = sit_acc
    elif index in lay:
        temp_acc = lay_acc

    index = index + 1
    img_id = img['id']
    img_path = os.path.join(args.data_path, img['file_name'])
    print(img_path)
    ret = detector.run(img_path)['results']
    ori_img = cv2.imread(img_path)
    pred_keypoints = []
    pred_bboxs = []
    for r in ret[1]:
        if r[4] > opt.vis_thresh:
            pt1 = (int(r[0]), int(r[1]))
            pt2 = (int(r[2]), int(r[3]))
            ori_img = cv2.rectangle(ori_img, pt1, pt2, (255, 0, 0), 2)
            pred_bboxs.append(r[:4])
            pred_keypoint = []
            for i in range(5, 39, 2):
                pred_keypoint.append((r[i], r[i+1]))
            pred_keypoints.append(pred_keypoint)
    try:
        gt = gts[img_id]
    except:
        continue
    for k in gt:
        gt_bbox = k['bbox']
        gt_keypoint = []
        gt_keypoint_mask = []
        for idx in range(len(k['keypoints'])):
            if idx % 3 == 0:
                if k['keypoints'][idx+2] == 0:
                    gt_keypoint_mask.append(-1)
                else:
                    gt_keypoint_mask.append(1)
                gt_keypoint.append((k['keypoints'][idx], k['keypoints'][idx+1]))
        if len(gt_keypoint) != 17:
            break
        if args.visualize:
            gt_keypoint[1:5][0] = gt_keypoint[0][0]
            gt_keypoint[1:5][1] = gt_keypoint[0][1]

            for i in range(17):
                ori_img = cv2.circle(ori_img, (int(float(gt_keypoint[i][0])), int(float(gt_keypoint[i][1]))), 2, (0, 255, 0), 2)
        np_gt_keypoint = np.array(gt_keypoint)
        np_pred_bboxs = np.array(pred_bboxs)
        np_gt_bbox = np.array(gt_bbox)
        np_gt_keypoint_mask = np.array(gt_keypoint_mask)
        # torso_dist = |left_shoulder - right_hip|, 1e-8 avoid zero division
        torso_dist = distance.euclidean(np_gt_keypoint[5], np_gt_keypoint[12]) + 1e-8
        if len(pred_bboxs) == 0:
            d = np.ones(17) * np.inf
            dist_keypoints.append(d)
        else:
            # greedly match human bounding box
            dist_bbox = np.sum(np.abs(np.subtract(pred_bboxs, gt_bbox)), axis=-1)
            target_id = int(np.argmin(dist_bbox))
            pred_keypoint = pred_keypoints[target_id]

            if args.visualize:
                for i in range(17):
                    ori_img = cv2.circle(ori_img, (int(pred_keypoint[i][0]), int(pred_keypoint[i][1])), 2, (0, 0, 255), 2)
            # calculate the keypoint distance and normalized by torso_dist
            d = np.sqrt(np.sum(np.square(np.subtract(gt_keypoint, pred_keypoint)), axis=-1)) / torso_dist
            d = d * np_gt_keypoint_mask
            dist_keypoints.append(d)
            temp_acc.append(d)
            del pred_keypoints[target_id]
            del pred_bboxs[target_id]
    # visualize
    if args.visualize:
        file_out_path = os.path.join('../output/', model_load_path)
        if not os.path.exists('../output/'):
            os.mkdir('../output/')
        if not os.path.exists(file_out_path):
            os.mkdir(file_out_path)

        cv2.imwrite(file_out_path + '/{}.jpg'.format(index), ori_img)
    # False Positive
    for _ in range(len(pred_bboxs)):
        d = np.ones(17) * np.inf
        dist_keypoints.append(d)
        temp_acc.append(d)
pck_correct = [0 for _ in range(17)]
num_samples = [len(dist_keypoints) for _ in range(17)]
for dist_keypoint in dist_keypoints:
    for idx, dist in enumerate(dist_keypoint):
        if dist < 0:
            num_samples[idx] -= 1
        elif dist <= args.threshold:
            pck_correct[idx] += 1

pck_correct = [p/n for p, n in zip(pck_correct, num_samples)]
print("Keypoints: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle")
print('PCK keypoints: {}' .format(pck_correct))
print('PCK all: {}' .format(sum(pck_correct)/len(pck_correct)))


getacc(stand_acc, 'stand')
getacc(sit_acc, 'sit')
getacc(lay_acc, 'lay')
#
# false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
# roc_auc = auc(false_positive_rate, true_positive_rate)
#
# plt.title('Receiver Operating Characteristic')
# plt.plot(false_positive_rate, true_positive_rate, 'b',
# label='AUC = %0.2f'% roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([-0.1,1.2])
# plt.ylim([-0.1,1.2])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig("filename.png")
#
# #stand:0:25, 77:84, 133:138
# #sit:25:39, 46:60, 72:77, 84:96
# #lay:39:46, 60:72, 96:133
