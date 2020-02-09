import os
import sys
import glob
import collections
import random
import multiprocessing as mp

def process(data):
    i, fn = data
    cmd = 'cp ' + fn + ' ' + os.path.join(out_fol, str(i + 1) + '.png')
    #print(i + 1, cmd)
    os.system(cmd)

#label_txt = 'clean_up_each_5000/sample_lepidic_5000/label.txt'
label_txt = 'clean_up_each_5000/sample_micropap_5000/label.txt'

ignores = set([f.split()[1] for f in open(label_txt, 'r')])

#classes = {'0':'clean_up_each_5000/sample_lepidic_6600'}
classes = {'3':'clean_up_each_5000/sample_micropap_additional_6000'}

for current_class in classes.keys():
    print('process class: ', classes[current_class])

    fns = [f for f in glob.glob('patches_lung_seer_john/*' + current_class + '.png')] +\
          [f for f in glob.glob('patches_lung_seer_john_additional/*' + current_class + '.png')]
    print(len(fns))
    maps = collections.defaultdict(list)

    cnt = 0
    slides_existed = collections.defaultdict(list)
    for fn in fns:
        #folder/001738-100046-multires.tif_62234_40913_576_400_0.png
        slide_id = fn.split('/')[-1].split('.')[0]
        if fn in ignores:
            cnt += 1
            slides_existed[slide_id].append(fn)
            continue
        maps[slide_id].append(fn)

    print('total number of files existed: ', cnt)
    print('number of existed slides: ', len(slides_existed))
    print('number of slides: ', len(maps))
    out = []
    for _, fn in maps.items():
        random.shuffle(fn)
        if len(fn) > 1000:
            fn = fn[:1000]
        numSample = len(fn)
        if len(fn) > 100:
            numSample = int(len(fn) / 2)

        out.extend(fn[:numSample])

    print('len of out: ', len(out))

    out_fol = classes[current_class]
    if not os.path.exists(out_fol):
        os.mkdir(out_fol)

    out = [(i, fn) for i, fn in enumerate(out)]
    pool = mp.Pool(processes = 64)
    pool.map(process, out)

    with open(out_fol + '/label.txt', 'w') as f:
        for i, fn in out:
            f.writelines('{} {}\n'.format(i + 1, fn))

