import numpy as np
import joblib
import cv2
import math
import multiprocessing
import time
import random

import numpy as np
from scipy.stats import skew


import numpy as np
import joblib
import cv2
from tqdm import tqdm


def subsample_dict(mydict, sample_percent = None):
	all_keys = list(mydict.keys())
	if sample_percent == None:
		sample_percent =0.1

	sample_size = int(sample_percent * len(all_keys))
	random_sampled_keys = random.sample(all_keys, sample_size)
	# print(random_sampled_keys)
	mydict = {key: mydict[key] for key in random_sampled_keys}
	return mydict

# def get_features(myimg, tile_preds):
# 	# saliency
# 	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
# 	(success, saliencyMap) = saliency.computeSaliency(myimg)

# 	output_list  = []

# 	for cellId in tile_preds:
# 		cell = tile_preds[cellId]
# 		contour = cell["contour"]

# 		# area
# 		area = cv2.contourArea(contour)

# 		# solidity
# 		hull = cv2.convexHull(contour)
# 		hull_area = cv2.contourArea(hull)
# 		solidity = float(area)/hull_area

# 		# major/minor axis len
# 		(x,y),(minAxisLen,majAxisLen),angle = cv2.fitEllipse(contour)
# 		perimeterLen = cv2.arcLength(contour, True)

# 		# circularity
# 		circularity = 4 * np.pi * area / (perimeterLen**2)

# 		# eccentiricity
# 		eccentiricity = math.sqrt((majAxisLen)**2 - (minAxisLen)**2) / (majAxisLen)

# 		# mean intensities
# 		mask = np.zeros(myimg[:,:,0].shape, np.uint8)
# 		cv2.drawContours(mask, [contour], -1, 255, -1)
# 		maskedimg = myimg[np.where(mask == 255)]
# 		intensities = np.mean(maskedimg, axis=0)

# 		intensities_var = np.var(maskedimg, axis=0)

# 		# intensity range
# 		mins = np.min(maskedimg, axis=0)
# 		maxes = np.max(maskedimg, axis=0)
# 		intensity_range = maxes - mins

# 		# boundary
# 		boundary = np.zeros(myimg[:,:,0].shape, np.uint8)
# 		# currently the boundary thickness is 3 pixels
# 		cv2.drawContours(boundary, [contour], -1, 255, 3)

# 		# boundary intensity
# 		boundaryimg = myimg[np.where(boundary == 255)]
# 		boundary_intensities = np.mean(boundaryimg, axis=0)

# 		# boundary saliency
# 		boundary_saliencies = saliencyMap[np.where(boundary == 255)]
# 		boundary_saliency_mean = np.mean(boundary_saliencies)

# 		output_list.append( intensity_range.tolist() + intensities.tolist() + boundary_intensities.tolist() + [area,eccentiricity, solidity,circularity, majAxisLen,minAxisLen, boundary_saliency_mean] )

# 		cell["area"] = area
# 		cell["eccentiricity"] = eccentiricity
# 		cell["solidity"] = solidity
# 		cell["circularity"] = circularity
# 		cell["majorAxisLength"] = majAxisLen
# 		cell["minAxisLength"] = minAxisLen
# 		cell["blue_mean_intensity"] = intensities[0]
# 		cell["green_mean_intensity"] = intensities[1]
# 		cell["red_mean_intensity"] = intensities[2]
# 		cell["blue_intensity_range"] = intensity_range[0]
# 		cell["green_intensity_range"] = intensity_range[1]
# 		cell["red_intensity_range"] = intensity_range[2]
# 		cell["blue_boundary_intensity"] = boundary_intensities[0]
# 		cell["green_boundary_intensity"] = boundary_intensities[1]
# 		cell["red_boundary_intensity"] = boundary_intensities[2]
# 		cell["boundary_saliency"] = boundary_saliency_mean

# 	return np.array(output_list)

def get_overall_statistics(features):
    overall_mean = np.mean(features, axis=0).tolist()
    overall_std = np.std(features, axis=0).tolist()
    overall_var = np.var(features, axis=0).tolist()
    overall_skewness = skew(features, axis=0).tolist()

    # Calculate quantiles at percentiles 0.1, 0.2, ..., 0.9
    quantiles = []
    for q in [10,25,75,90]: #range(0, 100, 10):
        quantiles += np.percentile(features, q, axis=0).tolist()

    result_list = overall_mean + overall_std + overall_var + overall_skewness + quantiles
    return result_list



def get_cell_features(celldata, image_global, saliencymap_global, original=False):
    contour_global = np.array(celldata['contour'])
    if original:
        contour = contour_global
        image = image_global
        saliencymap = saliencymap_global
    else:
        img_max_coord = image_global.shape[:2][::-1]
        # recalc from contour
        cont_min, cont_max = np.min(contour_global, axis=0), np.max(contour_global, axis=0)
        # add padding to allow proper saliency calc because boundary is 3 pixels
        padding = 5
        cont_min = np.clip(cont_min-padding, 0, img_max_coord)
        cont_max = np.clip(cont_max+padding, 0, img_max_coord)

        # inverted because array dims transpose of coordinates
        box_min, box_max = cont_min[::-1], cont_max[::-1]
        # filter cell if bbox is outside image dimensions
        if contour_global.shape[0] < 5:
            return None
        contour = contour_global - np.expand_dims(cont_min, axis=0)
        image = image_global[box_min[0]:box_max[0], box_min[1]:box_max[1], :]
        saliencymap = saliencymap_global[box_min[0]:box_max[0], box_min[1]:box_max[1]]

    # area
    area = cv2.contourArea(contour)
    # solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    # major/minor axis len
    (x,y),(minAxisLen,majAxisLen),angle = cv2.fitEllipse(contour)
    perimeterLen = cv2.arcLength(contour, True)
    # circularity
    circularity = 4 * np.pi * area / (perimeterLen**2)
    # eccentiricity
    eccentiricity = np.sqrt((majAxisLen)**2 - (minAxisLen)**2) / (majAxisLen)
    # mean intensities
    mask = np.zeros(image[:,:,0].shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    maskedimg = image[np.where(mask == 255)]
    intensities = np.mean(maskedimg, axis=0)
        
    # intensity range
    mins = np.min(maskedimg, axis=0)
    maxes = np.max(maskedimg, axis=0)
   
    intensity_range = maxes - mins

    # boundary
    boundary = np.zeros(image[:,:,0].shape, np.uint8)
    # currently the boundary thickness is 3 pixels
    cv2.drawContours(boundary, [contour], -1, 255, 3)

    # boundary intensity
    boundaryimg = image[np.where(boundary == 255)]
    boundary_intensities = np.mean(boundaryimg, axis=0)

    # boundary saliency
    boundary_saliencies = saliencymap[np.where(boundary == 255)]
    boundary_saliency_mean = np.mean(boundary_saliencies)

    # return {
    # 	"area" : area,
    # 	"eccentiricity" : eccentiricity,
    # 	"solidity" : solidity,
    # 	"circularity" : circularity,
    # 	"majorAxisLength" : majAxisLen,
    # 	"minAxisLength" : minAxisLen,
    # 	"blue_mean_intensity" : intensities[0],
    # 	"green_mean_intensity" : intensities[1],
    # 	"red_mean_intensity" : intensities[2],
    # 	"blue_intensity_range" : intensity_range[0],
    # 	"green_intensity_range" : intensity_range[1],
    # 	"red_intensity_range" : intensity_range[2],
    # 	"blue_boundary_intensity" : boundary_intensities[0],
    # 	"green_boundary_intensity" : boundary_intensities[1],
    # 	"red_boundary_intensity" : boundary_intensities[2],
    # 	"boundary_saliency" : boundary_saliency_mean,
    # }

    return np.array([
        area,
        eccentiricity,
        solidity,
        circularity,majAxisLen,
        minAxisLen,
        intensities[0],
        intensities[1],
        intensities[2],
        intensity_range[0],
        intensity_range[1],
        intensity_range[2],
        boundary_intensities[0],
        boundary_intensities[1],
        boundary_intensities[2],
        boundary_saliency_mean,
    ])

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
def get_saliency_map(image):
    _, saliencymap = saliency.computeSaliency(image)
    return saliencymap

def add_features_and_create_new_dicts(datpaths, imgpaths, updatpaths):
    for datpath, imgpath, updatpath in zip(tqdm(datpaths), imgpaths, updatpaths):
        celldata_dict = joblib.load(datpath)
        celldata_list = [(k, v) for k, v in celldata_dict.items()]
        # celldata_list = celldata_list[:100]

        image = cv2.imread(imgpath)
        saliencymap = get_saliency_map(image)

        cellfeat_list = joblib.Parallel(4)(
            joblib.delayed(get_cell_features)(celldata, image, saliencymap)
            for _, celldata in tqdm(celldata_list, disable=True)
        )
        # cellfeat_list = [
        #     get_cell_features(celldata, image, saliencymap)
        #     for _, celldata in tqdm(celldata_list, disable=True)
        # ]

        cellfeat_list_temp = [x for x in cellfeat_list if x is not None]
        valid_cells_n = len(cellfeat_list_temp)
        total_cells_n = len(cellfeat_list)
        if total_cells_n >= 100:
            valid_cell_frac = (valid_cells_n/total_cells_n)
            
            if valid_cell_frac < 1.0:
                print('updatpath:', updatpath)
                print('datpath:', datpath)
                print('imgpath:', imgpath)
                print(f'valid_cells_frac {valid_cell_frac} = 1-{total_cells_n-valid_cells_n}/{total_cells_n}')

            celldatanew_dict = {
                k: {**v, 'intensity_feats': feats}
                for (k, v), feats in zip(celldata_list, cellfeat_list)
                if feats is not None
            }
            joblib.dump(celldatanew_dict, updatpath)
        else:
             print("<100 cells", datpath, imgpath)
            