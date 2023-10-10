from seg import OvSegEasyuse
import glob

class_definition = {'building': [255,0,0], 
                    'plants':   [0,255,0]}
ose = OvSegEasyuse(class_definition)
for img_path in glob.glob('../autodl-tmp/stv_data/pics4/*/*.png'):
    ose.inference_and_save(img_path, 
                           img_path.replace('pics4', 'pics4-demo'),
                           img_path.replace('pics4', 'pics4-masked'))
    print(img_path)