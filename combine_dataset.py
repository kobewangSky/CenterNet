import os
import json

target_dirs = [ 'kinect_label_kobeF2', 'home_1', 'home_2', 'home_3', 'real_v2']
target_file = './data/'
for target_dir in target_dirs:
    target_file += target_dir + '_'
target_file += 'output.json'

output_images = {}
output_annotations = {}

for idx, target_dir in enumerate(target_dirs):
    target_json = os.path.join('./data', target_dir, 'annotations', 'output.json')
    labels = json.load(open(target_json))
    if idx == 0:
        output_images = labels['images']
        output_annotations = labels['annotations']
        for i in range(len(output_images)):
            output_images[i]['file_name'] = os.path.join(target_dir, 'images', output_images[i]['file_name'])
            output_images[i]['id'] = int(output_images[i]['id'])
        for i in range(len(output_annotations)):
            output_annotations[i]['image_id'] = int(output_annotations[i]['image_id'])
        print(len(output_images))
        print(len(output_annotations))
    else:
        temp_images = labels['images']
        temp_annotations = labels['annotations']
        for i in range(len(temp_images)):
            temp_images[i]['file_name'] = os.path.join(target_dir, 'images', temp_images[i]['file_name'])
            temp_images[i]['id'] = int(temp_images[i]['id']) + len(output_images)
        for i in range(len(temp_annotations)):
            temp_annotations[i]['image_id'] = int(temp_annotations[i]['image_id']) + len(output_images)
            temp_annotations[i]['id'] = len(output_images) + i
            # temp_annotations[i]['id'] = int(temp_annotations[i]['id']) + len(output_annotations)
        
        output_images.extend(temp_images)
        output_annotations.extend(temp_annotations)
        print(len(output_images))
        print(len(output_annotations))
output_json = {
    'images': output_images,
    'annotations': output_annotations
}

with open(target_file, 'w') as f:
    json.dump(output_json, f)


