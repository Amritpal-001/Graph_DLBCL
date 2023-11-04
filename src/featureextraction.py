import numpy as np
import joblib
import cv2
import math
import multiprocessing
import time
import random

import numpy as np
from scipy.stats import skew


def subsample_dict(mydict, sample_percent = None):
	all_keys = list(mydict.keys())
	if sample_percent == None:
		sample_percent =0.1

	sample_size = int(sample_percent * len(all_keys))
	random_sampled_keys = random.sample(all_keys, sample_size)
	# print(random_sampled_keys)
	mydict = {key: mydict[key] for key in random_sampled_keys}
	return mydict

def get_features(myimg, tile_preds):
	# saliency
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(myimg)

	output_list  = []

	for cellId in tile_preds:
		cell = tile_preds[cellId]
		contour = cell["contour"]

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
		eccentiricity = math.sqrt((majAxisLen)**2 - (minAxisLen)**2) / (majAxisLen)

		# mean intensities
		mask = np.zeros(myimg[:,:,0].shape, np.uint8)
		cv2.drawContours(mask, [contour], -1, 255, -1)
		maskedimg = myimg[np.where(mask == 255)]
		intensities = np.mean(maskedimg, axis=0)

		intensities_var = np.var(maskedimg, axis=0)

		# intensity range
		mins = np.min(maskedimg, axis=0)
		maxes = np.max(maskedimg, axis=0)
		intensity_range = maxes - mins

		# boundary
		boundary = np.zeros(myimg[:,:,0].shape, np.uint8)
		# currently the boundary thickness is 3 pixels
		cv2.drawContours(boundary, [contour], -1, 255, 3)

		# boundary intensity
		boundaryimg = myimg[np.where(boundary == 255)]
		boundary_intensities = np.mean(boundaryimg, axis=0)

		# boundary saliency
		boundary_saliencies = saliencyMap[np.where(boundary == 255)]
		boundary_saliency_mean = np.mean(boundary_saliencies)

		output_list.append( intensity_range.tolist() + intensities.tolist() + boundary_intensities.tolist() + [area,eccentiricity, solidity,circularity, majAxisLen,minAxisLen, boundary_saliency_mean] )

		cell["area"] = area
		cell["eccentiricity"] = eccentiricity
		cell["solidity"] = solidity
		cell["circularity"] = circularity
		cell["majorAxisLength"] = majAxisLen
		cell["minAxisLength"] = minAxisLen
		cell["blue_mean_intensity"] = intensities[0]
		cell["green_mean_intensity"] = intensities[1]
		cell["red_mean_intensity"] = intensities[2]
		cell["blue_intensity_range"] = intensity_range[0]
		cell["green_intensity_range"] = intensity_range[1]
		cell["red_intensity_range"] = intensity_range[2]
		cell["blue_boundary_intensity"] = boundary_intensities[0]
		cell["green_boundary_intensity"] = boundary_intensities[1]
		cell["red_boundary_intensity"] = boundary_intensities[2]
		cell["boundary_saliency"] = boundary_saliency_mean

	return np.array(output_list)

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