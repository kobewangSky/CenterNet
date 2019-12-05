# import os
# import json
#
# target_dirs = [ 'home_1', 'home_2', 'home_3', 'real_v0', 'real_v1', 'real_v2', 'real_v3', 'human_label_kobeF2', 'victor_1']
# target_file = './data/'
# for target_dir in target_dirs:
#     target_file += target_dir + '_'
# target_file += 'output.json'
#
# output_images = {}
# output_annotations = {}
#
# for idx, target_dir in enumerate(target_dirs):
#     target_json = os.path.join('./data', target_dir, 'annotations', 'output.json')
#     labels = json.load(open(target_json))
#     if idx == 0:
#         output_images = labels['images']
#         output_annotations = labels['annotations']
#         for i in range(len(output_images)):
#             output_images[i]['file_name'] = os.path.join(target_dir, 'images', output_images[i]['file_name'])
#             output_images[i]['id'] = int(output_images[i]['id'])
#         for i in range(len(output_annotations)):
#             output_annotations[i]['image_id'] = int(output_annotations[i]['image_id'])
#         print(len(output_images))
#         print(len(output_annotations))
#     else:
#         temp_images = labels['images']
#         temp_annotations = labels['annotations']
#         for i in range(len(temp_images)):
#             temp_images[i]['file_name'] = os.path.join(target_dir, 'images', temp_images[i]['file_name'])
#             temp_images[i]['id'] = int(temp_images[i]['id']) + len(output_images)
#         for i in range(len(temp_annotations)):
#             temp_annotations[i]['image_id'] = int(temp_annotations[i]['image_id']) + len(output_images)
#             temp_annotations[i]['id'] = len(output_images) + i
#             # temp_annotations[i]['id'] = int(temp_annotations[i]['id']) + len(output_annotations)
#
#         output_images.extend(temp_images)
#         output_annotations.extend(temp_annotations)
#         print(len(output_images))
#         print(len(output_annotations))
# output_json = {
#     'images': output_images,
#     'annotations': output_annotations
# }
#
# with open(target_file, 'w') as f:
#     json.dump(output_json, f)

import os
import json
import datetime
import numpy as np

IsPrivacy = True

if IsPrivacy:
    Privacyname = 'images_privacy'
else:
    Privacyname = 'images'
target_dirs = ['real_v0', 'real_v1', 'real_v2', 'real_v3', 'home_1', 'home_2', 'home_3', 'human_label_kobeF2', 'Virtual_V7', 'Virtual_V7_2', 'Virtual_V7_3', 'Virtual_V8_1', 'Virtual_victor_v1']
target_file = './data/'
target_file = target_file + Privacyname + '_'
for target_dir in target_dirs:
   target_file += target_dir + '_'

target_file += 'keypoint.json'

output_images = {}
output_annotations = {}

INFO = {
    "description": "Dataset",
    "url": "",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'human',
        'supercategory': 'human',
        'keypoints': ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                      "right_knee", "left_ankle", "right_ankle"],
        'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                     [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
    }
]

temp_id = 0
anotation_id = 0
for idx, target_dir in enumerate(target_dirs):
    target_json = os.path.join('./data', target_dir, 'annotations', 'output.json')
    labels = json.load(open(target_json))
    if idx == 0:
        max_id = 0
        output_images = labels['images']
        output_annotations = labels['annotations']
        for i in range(len(output_images)):
            output_images[i]['file_name'] = os.path.join(target_dir, Privacyname, output_images[i]['file_name'])
            output_images[i]['id'] = int(output_images[i]['id'])
            if output_images[i]['id'] > max_id:
                max_id = output_images[i]['id']
        for i in range(len(output_annotations)):
            output_annotations[i]['image_id'] = int(output_annotations[i]['image_id'])
            output_annotations[i]['id'] = '{}'.format(anotation_id)
            anotation_id = anotation_id + 1
        temp_id += max_id
    else:
        max_id = 0
        temp_images = labels['images']
        temp_annotations = labels['annotations']
        for i in range(len(temp_images)):
            temp_images[i]['file_name'] = os.path.join(target_dir, Privacyname, temp_images[i]['file_name'])
            temp_images[i]['id'] = int(temp_images[i]['id']) + temp_id
            if temp_images[i]['id'] > max_id:
                max_id = temp_images[i]['id']
        for i in range(len(temp_annotations)):
            temp_annotations[i]['image_id'] = int(temp_annotations[i]['image_id']) + temp_id
            temp_annotations[i]['id'] = '{}'.format(anotation_id)
            anotation_id = anotation_id + 1
            # temp_annotations[i]['id'] = int(temp_annotations[i]['id']) + len(output_annotations)

        output_images.extend(temp_images)
        output_annotations.extend(temp_annotations)
        temp_id += max_id

# check id is unique
image_ids = []
annotation_ids = []

for i in range(len(output_images)):
    image_ids.append(output_images[i]['id'])
for i in range(len(output_annotations)):
    annotation_ids.append(output_annotations[i]['id'])

image_ids = np.array(image_ids)
annotation_ids = np.array(annotation_ids)

unique = False
if len(image_ids) == len(np.unique(image_ids)):
    print('image_id is unique!')
    if len(annotation_ids) == len(np.unique(annotation_ids)):
        print('annotation_id is unique!')
        unique = True

# save file
output_json = {
    'info': INFO,
    'licenses': LICENSES,
    'categories': CATEGORIES,
    'images': output_images,
    'annotations': output_annotations
}

if unique:
    with open(target_file, 'w') as f:
        json.dump(output_json, f)
    print('save annotation!')



