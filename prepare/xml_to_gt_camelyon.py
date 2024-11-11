import os, sys
import numpy as np
import cv2
import openslide
import pdb

ind = sys.argv[1]
outd = sys.argv[2]
imgd = sys.argv[3]
reduce_scale = int(sys.argv[4]) #512 for x40, 256 for x20 (patch size 256 at x20)

os.makedirs(outd, exist_ok=True)

for root, dirs, files in os.walk(ind):
    for f in files:
        img = os.path.join(imgd, f.replace('.xml', '.tif'))
        gt = os.path.join(outd, f.replace('.xml', '.png'))
        if '.xml' in f and os.path.exists(img) and not os.path.exists(gt):
            slide = openslide.OpenSlide(img)
            w, h = slide.level_dimensions[0]
            if (w % reduce_scale) != 0 or (h % reduce_scale) != 0:
                print('non-integer multiple: ', f)
                continue

            mid_scale = reduce_scale // 32 # keep anno details, original img is too large to process
            resize_scale = reduce_scale // mid_scale
            out = np.zeros((h // mid_scale, w // mid_scale, 3)).astype('uint8') # patch level gt

            for cls in open(os.path.join(root, f), encoding='utf8', errors='ignore').read().split('Annotation Name="')[1:]:
                roi_contours = []

                for roi in cls.split('<Coordinates>')[1:]:
                    points = []

                    for p in roi.split(' X="')[1:]:
                        tks = p.split('" Y="')
                        points.append([int(float(tks[0]) / mid_scale + 0.5), int(float(tks[1].split('"')[0]) / mid_scale + 0.5)])

                    roi_contours.append(np.array(points))

                out = cv2.fillPoly(out, roi_contours, [0, 0, 1])

            # resize by keep max value (label=1 if 1 in the window)
            out = out[:, :, -1]
            out = out.reshape(out.shape[0] // resize_scale, resize_scale, out.shape[1] // resize_scale, resize_scale)
           
            # select max label as patch label
            out = out.max(1)
            out = out.max(2)
            
            ## ignore patch on the boundary
            #out = out.astype('float32')
            #out = out.mean(1)
            #out = out.mean(2)
            #out[out >= 0.5] = 1
            #out[(out > 0) * (out < 0.5)] = 255
            #out = out.astype('uint8')

            cv2.imwrite(os.path.join(outd, f.replace('.xml', '.png')), out)
            print(f)
