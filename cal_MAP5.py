## This file calculates MAP@5

import numpy as np 

def map_per_image(predictions, label):
    '''this function will calculate MAP@5 for a single image

    predictions = list of top 5 predictions for an image (Order does matter).
    label = true label
    '''
    try :
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError :
        return 0.0


def map_per_set(predictions, labels):
    '''this function calculates MAP@5 for the whole set

    predictions = list of list of top 5 predictions for every image.
    labels = list of true labels 
    '''

    return np.mean([map_per_image(p, l) for p,l in zip(predictions, labels)])