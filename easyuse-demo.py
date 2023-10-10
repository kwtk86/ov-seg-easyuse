from ov_seg_easyuse import OvSegEasyuse

# 定义字典，作为分割依据。
# 字典有多项，每项的键是需要分割的类型名称，值是保存结果时该类型对应的颜色
# 字典至少一项，至多254项
class_definition = {'building': [255,0,0], 
                    'plants':   [0,255,0],
                    'car':     [128,128,0]}
# 传入分割类型定义字典，调用OvSegEasyuse
ose = OvSegEasyuse(class_definition)

img_path = r'./demo_images/2339_118.3473081_34.35914428_180_201804.png'
out_seg_path = r'./demo_images_seg/2339_118.3473081_34.35914428_180_201804.png'
out_masked_path = r'./demo_images_masked/2339_118.3473081_34.35914428_180_201804.png'
# 对每张图像，调用ose.inference_and_save分割。
# inference_and_save有三个参数：img_path, out_seg_path, out_masked_img_path
# img_path: 输入图像的路径（必填）。
# out_seg_path: 输出的分割图路径。必填，后续分析一般基于这张图片及原图开展。
# out_masked_path输出的掩膜图。可选，可用于了解分割结果是否正确，分析一般不基于这张图片展开。
# 可以查看对应文件夹中的示例图片了解更多。
# 如果有多张图像，可以自行采取目录遍历等方式进行分割，只要调用inference_and_save就可以。
# inference_and_save不仅保存分割结果，也可以返回分割结果。用户可以根据需求对分割结果进行处理。

seg = ose.inference_and_save(img_path, out_seg_path, out_masked_path)
print('finish')