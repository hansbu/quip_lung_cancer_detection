import os
import sys
import numpy as np
import cv2
from PIL import Image
import openslide
import random
import multiprocessing as mp
import time
import pdb

'''
color_codes = {'NSCLC-Lapidic':(255, 0, 0), 'NSCLC-Benign':(255, 127, 0),
                'NSCLC-Adeno CA (all)':(255, 255, 0), 'NSCLC-Solid':(0, 255, 0),
                'NSCLC-Acinar':(0, 0, 255), 'NSCLC-Micropapillary':(255, 255, 255),
                'NSCLC-Papillary':(139, 0, 255)}
class_ids = {'NSCLC-Lapidic':0, 'NSCLC-Benign':1,
             'NSCLC-Adeno CA (all)':2, 'NSCLC-Solid':3,
             'NSCLC-Acinar':4, 'NSCLC-Micropapillary':5,
             'NSCLC-Papillary':6}
'''

classes = ['NSCLC-Lapidic', 'NSCLC-Benign', 'NSCLC-Acinar', 'NSCLC-Micropapillary', 'dummy', 'NSCLC-Adeno CA (all)', 'NSCLC-Solid']
class_ids = {classes[i]:i for i in range(len(classes))}

color_codes = {'NSCLC-Lapidic':(255, 0, 0), 'NSCLC-Benign':(255, 127, 0),
            'NSCLC-Adeno CA (all)':(255, 255, 0), 'NSCLC-Solid':(0, 255, 0),
            'NSCLC-Acinar':(0, 0, 255), 'NSCLC-Micropapillary':(255, 255, 255)}

input_mask_fol = 'json_to_image'
svs_fol = '/data10/shared/hanle/svs_SEER_Lung'
svs_fol2 = '/data04/shared/hanle/lung_cancer_prediction_seer_rutger_3/data/svs'
svs_fol3 = '/data02/erich/rutgers/newlung'
svs_fols = [svs_fol, svs_fol2, svs_fol3]

def isFileExists(folder, slideID):
    slide_path = os.path.join(folder, slideID)
    if os.path.exists(slide_path):
        return True
    return False

def find_blobs(img, xScale=1.0, yScale=1.0):
    print('shape of img: ', img.shape)
    img = img*255
    img = img.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    # find contours in the binary image
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('len of contours: ', len(cnts))
    if len(cnts) == 3:
        _, contours, _ = cnts
    else:
        contours, _ = cnts
    out = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"] * xScale)
            cY = int(M["m01"] / M["m00"] * yScale)
            out.append((cY, cX))
    return out

out_fol = 'corrs_to_extract'
svs_extension = 'tif'

slide_ids = [fn for fn in os.listdir(input_mask_fol) if '.png' in fn]
patch_size_20X = 400

def extract_patch(slide):
    annots_path = os.path.join(input_mask_fol, slide)
    annots = np.array(Image.open(annots_path).convert('RGB'))
    masks = {}
    for key, _ in color_codes.items():
        masks[key] = np.zeros(annots.shape[:2])

    slideID = slide[:-4] + '.' + svs_extension

    slide_path = None
    for fol in svs_fols:
        if isFileExists(fol, slideID):
            slide_path = os.path.join(fol, slideID)
            break
    if slide_path is None:
        print('file not found: ', slideID)
        return

    oslide = openslide.OpenSlide(slide_path)
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
    elif "XResolution" in oslide.properties:
        mag = 10.0 / float(oslide.properties["XResolution"])
    elif 'tiff.XResolution' in oslide.properties:  # for Multiplex IHC WSIs, .tiff images
        Xres = float(oslide.properties["tiff.XResolution"])
        if Xres < 10:
            mag = 10.0 / Xres
        else:
            mag = 10.0 / (10000 / Xres)  # SEER PRAD
    else:
        print('[WARNING] mpp value not found. Assuming it is 40X with mpp=0.254!', slide)
        mag = 10.0 / float(0.254);

    width = oslide.dimensions[0];
    height = oslide.dimensions[1];
    if abs(height / annots.shape[0] - width / annots.shape[1]) > 0.1:
        print('============================ERROR=================')
        return

    R, G, B = annots[:, :, 0], annots[:, :, 1], annots[:, :, 2]
    for key in masks.keys():
        cond = np.logical_and(R == color_codes[key][0], G == color_codes[key][1])
        cond = np.logical_and(cond, B == color_codes[key][2])
        masks[key][cond] = 1
        print(key, class_ids[key], color_codes[key], np.sum(masks[key]))
        sys.stdout.flush()
    #pdb.set_trace()

    corrs = {}
    for key in color_codes.keys():
        corrs[key] = []

    scale = height / annots.shape[0]
    pw_mask = int(patch_size_20X * mag / 20 / scale)  # scale patch size from 20X to 'mag'

    threshold = 0.75
    for r in range(0, annots.shape[0] - pw_mask, pw_mask):
        for c in range(0, annots.shape[1] - pw_mask, pw_mask):
            for key in corrs.keys():
                if np.sum(masks[key][r:r + pw_mask, c:c + pw_mask]) > threshold * pw_mask * pw_mask:
                    corrs[key].append((int(r * scale), int(c * scale)))

    for key, val in corrs.items():
        random.shuffle(val)
        limit = 500
        corrs[key] = val[:min(limit, len(val))]

    for key in corrs.keys():
        if np.sum(masks[key]) > 50:
            blob_y_x = find_blobs(masks[key], scale, scale)
            corrs[key].extend(blob_y_x)

    corr_file = open(os.path.join(out_fol, slide[:-4] + '.' + svs_extension + '.txt'), 'w')
    for key, val in corrs.items():
        for y, x in val:
            corr_file.writelines('{} {} {} {}\n'.format(slide.split('.')[0], x, y, class_ids[key]))

    corr_file.close()

start = time.time()
pool = mp.Pool(processes=16)
pool.map(extract_patch, slide_ids)
print('Elapsed time: ', (time.time() - start) / 60.0)
