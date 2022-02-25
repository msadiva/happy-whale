## importing libraries
import fastai
from fastai.vision.all import *
from icevision.models import *
from icevision.all import *
from mmcv.runner import (
    load_checkpoint,
    save_checkpoint,
    _load_checkpoint,
    load_state_dict,
)
import cv2
import imagesize
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import json
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width',1000)

## importing dataframe
fluke = pd.read_csv("keypoints.csv", low_memory = False)

## function to get image size
def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['Image'])
    return row

np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255))\
          for idx in range(1)]

## function to get coordinates
def map_keypoints(points):
    keypoints = []
    for i in range(len(points) // 2):
        keypoints.append((points[2*i], points[(2*i) + 1]))
    return keypoints

## get coordinates
points = fluke.iloc[:, 1:]
all_points = []
for i in points.values :
    key = map_keypoints(i)
    all_points.append(key)

fluke["all_points"] = all_points

## function to get bounding boxes
def bounding_rectangle(points):
    x0, y0 = points[0]
    x1, y1 = x0, y0
    for x,y in points[1:] :
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    bbox = [(x0, y0), (x1, y1)]
    return bbox

fluke["bbox"] = fluke["all_points"].apply(bounding_rectangle)

## get imagesize
fluke = fluke.rename(columns = {'filename' : 'Image'})
fluke = fluke.apply(get_imgsize, axis = 1)

## get xmin, ymin, xmax and ymax
final_df = fluke[["Image", "bbox", "width", "height"]]
final_df["xmin"] = final_df["bbox"].apply(lambda x: x[0][0])
final_df["ymin"] = final_df["bbox"].apply(lambda x: x[0][1])
final_df["xmax"] = final_df["bbox"].apply(lambda x: x[1][0])
final_df["ymax"] = final_df["bbox"].apply(lambda x: x[1][1])

## getting the label
final_df["label"] = "whale"

## save the final dataframe
final_df.to_csv("train-annotations.csv", index = False)
print (f"Final dataframe saved to directory {Path.cwd()}/train-annotations.csv")