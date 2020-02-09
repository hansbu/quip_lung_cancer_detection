import json
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import glob
import collections
import openslide
import os
import pdb

'''
color_codes = {'NSCLC-Lapidic':(255, 0, 0), 'NSCLC-Benign':(255, 127, 0),
                'NSCLC-Adeno CA (all)':(255, 255, 0), 'NSCLC-Solid':(0, 255, 0),
                'NSCLC-Acinar':(0, 0, 255), 'NSCLC-Micropapillary':(255, 255, 255),
                'NSCLC-Papillary':(139, 0, 255)}

'''
#classes = ['NSCLC-Lapidic', 'NSCLC-Benign', 'NSCLC-Acinar', 'NSCLC-Micropapillary', 'NSCLC-Papillary']
#colors = [(255, 0, 0), (255, 127, 0), (0, 0, 255), (255, 255, 255), (139, 0, 255)]
#color_codes = {classes[i]:colors[i] for i in range(len(colors))}

#color_codes = {'NSCLC-Micropapillary':(255, 255, 255)}
color_codes = {'NSCLC-Lapidic':(255, 0, 0), 'NSCLC-Benign':(255, 127, 0),
                'NSCLC-Adeno CA (all)':(255, 255, 0), 'NSCLC-Solid':(0, 255, 0),
                'NSCLC-Acinar':(0, 0, 255), 'NSCLC-Micropapillary':(255, 255, 255)}

annots_fol = 'SEER-Rutgers-Lung-2020-2-3-2-32-49'

annotated_slides = {s.rstrip() for s in open(os.path.join(annots_fol, 'annotated.txt'), 'r')}

annot_types = collections.defaultdict(int)
manifest = [f.rstrip().split(',') for f in open(annots_fol + '/manifest.csv')]
fn_to_slideID = {fn.replace("\"", ""):slideID.split('/')[-1].replace("\"", "") for _, _, _, _, _, slideID, fn in manifest[1:]}
svs_fol = '/data10/shared/hanle/svs_SEER_Lung'
svs_fol2 = '/data04/shared/hanle/lung_cancer_prediction_seer_rutger_3/data/svs'
svs_fol3 = '/data02/erich/rutgers/newlung'
svs_fols = [svs_fol, svs_fol2, svs_fol3]

def isFileExists(folder, slideID):
    slide_path = os.path.join(folder, slideID)
    if os.path.exists(slide_path):
        return True
    return False

john_creator = '49'
notes = collections.defaultdict(int)
numWSINotFound = 0
for i, fn in enumerate(glob.glob(annots_fol + '/*.json')):
    slideID = fn_to_slideID[fn.split('/')[-1]]
    if slideID[:13] not in annotated_slides:
        print('Slide is not annotated')
        continue

    slide_path = None
    for fol in svs_fols:
        if isFileExists(fol, slideID):
            slide_path = os.path.join(fol, slideID)
            break
    if slide_path is None:
        print('WSI not found: ================', slideID)
        numWSINotFound += 1
        continue

    oslide = openslide.OpenSlide(slide_path)
    width, height = oslide.level_dimensions[5]      # extract the width and height at level 5
    #slide_path = slideID
    #width, height = 400, 400

    print(i, fn, slide_path, width, height)

    data = json.load(open(fn))
    if len(data) == 0:
        continue

    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    numValidRegions = 0
    for region in data:
        if 'creator' not in region:
            continue

        creator = region['creator']
        if creator != john_creator:
            continue

        coors = region['geometries']['features'][0]['geometry']['coordinates'][0]
        coors_converted = [(int(x*width), int(y*height)) for x, y in coors]
        if 'notes' not in region['properties']['annotations']: continue
        annot_type = region['properties']['annotations']['notes']

        if annot_type not in color_codes: continue
        draw.polygon(coors_converted, fill=color_codes[annot_type])
        numValidRegions += 1
        notes[annot_type] += 1

    if numValidRegions > 0:
        image.save('json_to_image/' + slide_path.split('/')[-1][:-4] + '.png')

print(notes)
print('number of WSI not found: ', numWSINotFound)
