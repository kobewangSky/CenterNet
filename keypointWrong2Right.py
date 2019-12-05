import json
from glob import glob
import cv2
import os
import sys
import time
import argparse
import shutil
import numpy as np
from pycocotools.coco import COCO
import math
##由於label 工具label出來的id 是用string表示，是錯的，應該要用int表示
license_url = ["http://creativecommons.org/licenses/by-nc-sa/2.0/",
               "http://creativecommons.org/licenses/by-nc/2.0/",
               "http://creativecommons.org/licenses/by-nc-nd/2.0/",
               "http://creativecommons.org/licenses/by/2.0/",
               "http://creativecommons.org/licenses/by-sa/2.0/",
               "http://creativecommons.org/licenses/by-nd/2.0/",
               "http://flickr.com/commons/usage/",
               "http://www.usa.gov/copyright.shtml"]

license_name = ["Attribution-NonCommercial-ShareAlike License",
                "Attribution-NonCommercial License",
                "Attribution-NonCommercial-NoDerivs License",
                "Attribution License",
                "Attribution-ShareAlike License",
                "Attribution-NoDerivs License",
                "No known copyright restrictions",
                "United States Government Work"
                ]


def createAnnotateData(id, category_id, iscrowd, img_id, bbox, keypointlist):
    annotate_info = {}
    annotate_info["id"] = id
    annotate_info["image_id"] = img_id
    annotate_info["category_id"] = category_id
    annotate_info["bbox"] = bbox
    annotate_info["iscrowd"] = iscrowd

    area = math.sqrt(bbox[2] * bbox[3]**2)

    annotate_info["area"] = area
    annotate_info["keypoints"] = keypointlist
    return annotate_info


def createImaData(id, width, height, filename, license, flicker_url, coco_url, date_captured):
    img_info = {}
    img_info["id"] = id
    img_info["width"] = width
    img_info["height"] = height
    img_info["file_name"] = filename
    img_info["license"] = license
    img_info["flicker_url"] = flicker_url
    img_info["coco_url"] = coco_url
    img_info["date_captured"] = date_captured
    return img_info


def createCategorieData(id, name, supercategory):
    categories_data = {}
    categories_data["id"] = id
    categories_data["name"] = name
    categories_data["supercategory"] = supercategory
    categories_data["keypoints"] = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                                    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                                    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    categories_data["skeleton"] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                                   [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    categories_data["keypoints_style"] = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0",
                                          "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8",
                                          "#800000", "#aaffc3", "#808000"]

    return categories_data


def createEmptyDataSet(description, url, version, year, contributor, date):
    dataset = {}
    # info
    dataset["info"] = {}
    dataset["info"]["description"] = description
    dataset["info"]["url"] = url
    dataset["info"]["version"] = version
    dataset["info"]["year"] = year
    dataset["info"]["contributor"] = contributor
    dataset["info"]["data_created"] = date
    # licenses
    dataset["licenses"] = []
    for idx, url in enumerate(license_url):
        license = {}
        license["url"] = url
        license["id"] = idx
        license["name"] = license_name[idx]
        dataset["licenses"].append(license)
    # images
    dataset["images"] = []
    # annotations
    dataset["annotations"] = []
    # categories
    dataset["categories"] = []
    return dataset


def collectkeypoint(obj):

    keylist = [3, 3, 3, 3, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]

    output = []
    for obj_it in keylist:
        output.append(float(obj[obj_it][0]))
        output.append(float(obj[obj_it][1]))
        output.append(2)
    return output


def getCidByname(cs, name):
    id = -1
    for c in cs:
        if c["name"] == name:
            id = c["id"]
            break
    return id


def getSegColorByName(obj_setting, name):
    segColor = 0
    for obj in obj_setting["exported_objects"]:
        if obj["class"] == name:
            return obj["segmentation_class_id"]
    return segColor


def mainProcess(RawDir, outputDir):


    coco = COCO(RawDir)

    print("--------step1:  create empty dataset ------------")
    dataset = createEmptyDataSet("MMTD DataSet (COCO Style)", "", "1.0", 2019, "CGV Team",
                                 time.strftime("%Y/%m/%d", time.localtime()))

    ImageCopyList = 0
    AnnotImageList = 0

    Annotate_id = 0

    begin = time.time()

    inputDir = RawDir

    # process category data
    print("--------step2:  process category ---------------")

    cdata = createCategorieData(1, 'person', 'person')  # keypoint just have person as 1
    dataset["categories"].append(cdata)


    # process image data
    print("--------step3:  process imageinfo --------------")
    # imgSet = glob(inputDir + '/img' + '/[0-9]*.png')
    #     # jsonSet = glob(inputDir + '/jnt' + '/[0-9]*.txt')
    #     # imgSet.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #     # jsonSet.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #     # imgs = []



    for idx, imgindx in enumerate(coco.getImgIds()):
        imgdata = coco.loadImgs([imgindx])



        img_data = createImaData(int(imgdata[0]['id']), imgdata[0]['width'], imgdata[0]['height'], imgdata[0]['file_name'], imgdata[0]['license'], imgdata[0]['flicker_url'], imgdata[0]['coco_url'], imgdata[0]['date_captured'])
        dataset["images"].append(img_data)
        ImageCopyList = ImageCopyList + 1

    # process annotate data
    print("--------step4:  process annotations -----------")

    for idx, Annindx in enumerate(coco.getAnnIds()):
        Anndata = coco.loadAnns([Annindx])

        # print(AnnotImageList)
        # obj_json = open(jsonSet[idx], "r")
        # data = obj_json.read()
        # data = data.split('\n')
        # temp_data = []
        # for joint in data[:-1]:
        #     joint = joint[:-1]
        #     point = joint.split(' ')
        #     temp_data.append(point)
        # np_temp_data = np.array(temp_data)
        #
        # xmin = float(min(np_temp_data[:, 0]))
        # ymin = float(min(np_temp_data[:, 1]))
        #
        # xmax = float(max(np_temp_data[:, 0]))
        # ymax = float(max(np_temp_data[:, 1]))
        # # create class list
        #
        # pre_bbox = getBoundingBox(img.shape[0], img.shape[1], ymin, xmin, ymax, xmax)
        # # top-left-x , top-left-y, w, h
        # bbox = [pre_bbox[1], pre_bbox[0], pre_bbox[3] - pre_bbox[1], pre_bbox[2] - pre_bbox[0]]
        # bbox = [round(i, 2) for i in bbox]
        # # get segmentation info
        #
        # keypointoutput = collectkeypoint(np_temp_data)

        annotate_data = createAnnotateData(idx, 1, 0, int(Anndata[0]['image_id']), Anndata[0]['bbox'], Anndata[0]['keypoints'])

        dataset["annotations"].append(annotate_data)
        Annotate_id = Annotate_id + 1

        AnnotImageList = AnnotImageList + 1
    print("--------step4:  process execute time ----------")
    print(time.time() - begin)


    with open(outputDir, 'w') as outfile:
        json.dump(dataset, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--inputDir", help="input path need to process", default=".", dest="inputDir")
    parser.add_argument("-o", "--outputDir", help="output path", default=".", dest="outputDir")
    args = parser.parse_args()
    mainProcess(args.inputDir, args.outputDir)