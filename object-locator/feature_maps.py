
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
from models import unet_model_regress_px
from collections import OrderedDict
import torch.nn.functional as F
import math
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import os
import pandas as pd
from matplotlib.patches import Rectangle, Circle

# Used this tutorial: https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573 (this is wrong, don't use!)
# and also this tutorial: https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
# used this forum post: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6

#### Inputs
input_dir = '/Users/Zahra1/Documents/Stanford/Research/datasets/SEDC/64x64/images_64/'
# imgName   = 'sister_R04_v2_rez2_snr2_0615_0800_nm_r2_64px.png'
imgName   = 'sister_R05_v2_sez1_snr2_0425_0552_nm_r2_64px.png'
inputImg = os.path.join(input_dir,imgName)
chkpt = '/Users/Zahra1/Documents/Stanford/Research/locating-objects-without-bboxes/PSR_200ep_16b_05tau_1r_split7.ckpt'

#### Define transform
transform = transforms.Compose([
                                transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),
                                                     (0.5,0.5,0.5)),
                               ])
#### Load image
image = Image.open(inputImg)
# plt.imshow(image)
# plt.show()

#### CUDA or CPU
torch.set_default_dtype(torch.float32)
device_cpu = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else device_cpu

#### Create model and load checkpoint
model = unet_model_regress_px.UNet_px(3,1,
                                         height=64,
                                         width=64,
                                         device=device,
                                         ultrasmall=True)
model = model.to(device)
# Load checkpoint
checkpoint = torch.load(chkpt, map_location=lambda storage, loc: storage)

state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:]
    state_dict[name] = v
model.load_state_dict(state_dict)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)

#### Prepare input image
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

#### Define Forward Hook
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#### Attach forward hooks on all convolution layers
model.inc.conv.conv[0].register_forward_hook(get_activation('inconv_conv1'))
model.inc.conv.conv[3].register_forward_hook(get_activation('inconv_conv2'))
model.down1.mpconv[1].conv[0].register_forward_hook(get_activation('down1_conv1'))
model.down1.mpconv[1].conv[3].register_forward_hook(get_activation('down1_conv2'))
model.down2.mpconv[1].conv[0].register_forward_hook(get_activation('down2_conv1'))
model.down2.mpconv[1].conv[3].register_forward_hook(get_activation('down2_conv2'))
model.down3.mpconv[1].conv[0].register_forward_hook(get_activation('down3_conv1'))
model.down3.mpconv[1].conv[2].register_forward_hook(get_activation('down3_conv2'))
model.up1.conv.conv[0].register_forward_hook(get_activation('up1_conv1'))
model.up1.conv.conv[3].register_forward_hook(get_activation('up1_conv2'))
model.up2.conv.conv[0].register_forward_hook(get_activation('up2_conv1'))
model.up2.conv.conv[3].register_forward_hook(get_activation('up2_conv2'))
model.up3.conv.conv[0].register_forward_hook(get_activation('up3_conv1'))
model.up3.conv.conv[3].register_forward_hook(get_activation('up3_conv2'))
model.outc.conv.register_forward_hook(get_activation('outconv'))

output = model(image)

# idxs = ['inconv_conv1','inconv_conv2','down1_conv1','down1_conv2','down2_conv1',
#         'down2_conv2','down3_conv1','down3_conv2','up1_conv1','up1_conv2','up2_conv1',
#         'up2_conv2','up3_conv1','up3_conv2','outconv']
idxs = ['inconv_conv1','down2_conv1','down3_conv2','up1_conv2','up3_conv2']

for layer in activation:
    print(activation[layer].shape)

### Plotting 5 feature maps for a handful of convolution layers
# current = 1
# fig2 = plt.figure()
# for layer in idxs:
#     featmap = activation[layer].squeeze(0).cpu().numpy()
#     for j in range((5*(current-1)),5*current):
#         a = fig2.add_subplot(5,5,j+1)
#         imgplot = plt.imshow(featmap[j+20,:,:])
#         a.axis("off")
#     current += 1
# fig2.tight_layout()
# plt.subplot_tool()
# plt.show()

### Use the 50th feature map for R5 and 10th for R4

current = 1
fig2 = plt.figure()
for layer in idxs:
    featmap = activation[layer].squeeze(0).cpu().numpy()
    a = fig2.add_subplot(1,5,current)
    if layer == 'outconv':
        imgplot = plt.imshow(featmap[0,:,:])
    else:
        imgplot = plt.imshow(featmap[50,:,:])
    a.axis("off")
    current += 1
fig2.tight_layout()
# plt.subplot_tool()
plt.subplots_adjust(left=0.023,
                    bottom=0.12,
                    right=0.977,
                    top=0.871,
                    wspace=0.05,
                    hspace=0.023)
plt.show()



#### Save out all conv layers
# model_weights =[]
# conv_layers = []
# # function to recursively go through all modules
# def get_conv_layers_weights(module):
#     if isinstance(module, nn.Conv2d):
#         conv_layers.append(module)
#         model_weights.append(module.weight)
#     else:
#         for child in module.children():
#             get_conv_layers_weights(child)

# get_conv_layers_weights(model)
# print(f"Total convolution layers: {len(conv_layers)}")
# print(conv_layers)

# image = transform(image)
# print(f"Image shape before: {image.shape}")
# image = image.unsqueeze(0)
# print(f"Image shape after: {image.shape}")
# image = image.to(device)
# original_image = image.to(device)

# # Pass input through all conv layers
# outputs = []
# names = []
# up = nn.Upsample(scale_factor=2,
#                               mode='bilinear',
#                               align_corners=True)
# concat = np.sort(np.arange(1,14))[::-1]
# for i in range(len(conv_layers)):
#     layer = conv_layers[i]
#     if image.shape[1] == layer.in_channels:
#         image = layer(image)
#         outputs.append(image)
#         names.append(str(layer))
#     elif image.shape[1] < layer.in_channels:
#         x1 = up(image)
#         x2_idx = concat[i]
#         x2 = outputs[x2_idx]
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2)),
#                         diffY // 2, int(math.ceil(diffY / 2))))
#         x = torch.cat([x2, x1], dim=1)
#         image = layer(x)
#         outputs.append(image)
#         names.append(str(layer))
#         continue
# print(len(outputs))
# #print feature_maps
# for feature_map in outputs:
#     print(feature_map.shape)

#### Summing all feature maps for each layer -- Don't use this
# processed = []
# for feature_map in outputs:
#     feature_map = feature_map.squeeze(0)
#     gray_scale = torch.sum(feature_map,0)
#     gray_scale = gray_scale / feature_map.shape[0]
#     processed.append(gray_scale.data.cpu().numpy())
# for fm in processed:
#     print(fm.shape)

#### Plot the feature maps summed for each convolution layer
# fig = plt.figure()
# for i in range(len(processed)):
#     a = fig.add_subplot(3, 5, i+1)
#     imgplot = plt.imshow(processed[i])
#     a.axis("off")
# fig.tight_layout()
# plt.subplots_adjust(left=0.064,
#                     bottom=0.133,
#                     right=0.943,
#                     top=0.983,
#                     wspace=0,
#                     hspace=0.021)
# plt.show()

#### Plots showing a single filter for each convolution layer (i.e. not added)
# fig = plt.figure()
# for i in range(len(outputs)-1):
#     a = fig.add_subplot(3, 5, i+1)
#     ftmap = outputs[i].squeeze(0).data.cpu().numpy()
#     imgplot = plt.imshow(ftmap[18,:,:])
#     a.axis("off")
# fig.tight_layout()
# plt.subplots_adjust(left=0.064,
#                     bottom=0.133,
#                     right=0.943,
#                     top=0.983,
#                     wspace=0,
#                     hspace=0.021)
# plt.show()

#### Plots for the filters of a single convolution layer (i.e. not added)
# fig2 = plt.figure()
# for j in range(64):
#     feature_map = outputs[8].squeeze(0).data.cpu().numpy()
#     a = fig2.add_subplot(8,8,j+1)
#     imgplot = plt.imshow(feature_map[j,:,:])
#     a.axis("off")
# fig2.tight_layout()
# plt.show()

#### Heat Map
# out = model(original_image)
# conf_map = out[:,:,:,0]
# conf_map = est_map_np = torch.sigmoid(conf_map[0, :, :]).to(device_cpu).detach().numpy()
# output_3d = np.repeat(conf_map[:, :,np.newaxis], 3, axis=2)
# original_image_float = original_image.numpy().squeeze(0).astype(float).transpose((1,2,0))
# blended = original_image_float*0.7 + output_3d*0.3
# plt.imshow(conf_map)
# plt.colorbar()
# plt.show()







