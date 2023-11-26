#!pip install git+https://github.com/facebookresearch/segment-anything.git
import gdown
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files 

class SAM:
    def __init__(self):
        url = 'https://drive.google.com/uc?id=1T8hLqIvE-_i4oksmldr5b8G28XvCWxoo'
        output = '/content/'
        gdown.download(url, output, quiet=False)
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        
    def perfrom_segmentation(self, source, pps, pit, sst, resize):
        image = cv2.imread(source)
        mask_generator_ = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=pps,
            pred_iou_thresh=pit,
            stability_score_thresh=sst,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=1,
        )

        if resize[0]:
            image = cv2.resize(image, (resize[1], resize[2]))

        masks = mask_generator_.generate(image)

        def show_anns(anns):
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)

            img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 1
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([[1,1,1], [1]])
                img[m] = color_mask

            return img

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.imshow(show_anns(masks))
        plt.axis('off')
        plt.savefig('segmented_image.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
        files.download('segmented_image.png')

    def run(self, points_per_side=50, pred_iou_thresh=0.90, stability_score_thresh=0.90, resize=(True, 256, 256)):
        print('\n\n===================================================\n')
        print('Upload the image')
        source = files.upload()
        source_filename = list(source.keys())[0]
        print(f'File "{source_filename}" uploaded successfully!')
        print('Performing Segmentation')
        self.perfrom_segmentation(source_filename, points_per_side, pred_iou_thresh, stability_score_thresh, resize)
        
