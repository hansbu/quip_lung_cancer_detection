import os
import sys
import glob
import collections
import random
import multiprocessing as mp

def process(data):
    i, fn = data
    cmd = 'cp ' + fn + ' ' + os.path.join(out_fol, str(i + 1) + '.png')
    #print(i, cmd)
    os.system(cmd)

classes = {'0':'sample_lapidic_5000', '2':'sample_acinar_5000', '3':'sample_micropap_5000'}
samples_per_slide = {'0':170, '2':22, '3':138}
for current_class in classes.keys():
    print('process class: ', classes[current_class])
    number_patches_per_slide = samples_per_slide[current_class]

    fns = [f for f in glob.glob('patches_lung_seer_john/*' + current_class + '.png')] +\
          [f for f in glob.glob('patches_lung_seer_john_additional/*' + current_class + '.png')]
    print(len(fns))
    maps = collections.defaultdict(list)

    for fn in fns:
        #folder/001738-100046-multires.tif_62234_40913_576_400_0.png
        slide_id = fn.split('/')[-1].split('.')[0]
        maps[slide_id].append(fn)

    print('number of slides: ', len(maps))
    out = []
    for _, fn in maps.items():
        random.shuffle(fn)
        out.extend(fn[:number_patches_per_slide])
    print('len of out: ', len(out))

    out_fol = classes[current_class]
    if not os.path.exists(out_fol):
        os.mkdir(out_fol)

    out = [(i, fn) for i, fn in enumerate(out)]
    pool = mp.Pool(processes = 32)
    pool.map(process, out)

    with open(out_fol + '/label.txt', 'w') as f:
        for i, fn in out:
            f.writelines('{} {}\n'.format(i + 1, fn))

