import cv2
import numpy as np
from rembg import remove

# %%
image_source = "./source.jpg"
image_save = "./image.png"
mask_save = "./mask.png"
model_save = "./model.obj"    

# %%
img = cv2.imread(image_source,cv2.IMREAD_COLOR)
img = remove(img)
img = cv2.resize(img,(int(img.shape[1]/img.shape[0]*512),512))

# %%
def GenerateMask(img):
    mask = (img[:,:,3] > 0) * np.uint8(1)
    mask = cv2.erode(mask,  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3),(1,1)))
    mask = cv2.dilate(mask,  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3),(1,1)))
    return mask
    
def Padding(size,img):
    if(len(img.shape) > 2):
        size = (size[0],size[1],img.shape[2])
    ret = np.zeros(size,dtype=np.uint8)
    ratio = img.shape[0]/img.shape[1]
    if(ratio > 0):
        img = cv2.resize(img,(int(ret.shape[0]/ratio),ret.shape[0]))
    else:
        img = cv2.resize(img,(ret.shape[1],int(ret.shape[1]*ratio)))
    c_y=ret.shape[0]//2
    c_x=ret.shape[1]//2
    _c_y = img.shape[0]//2
    _c_x = img.shape[1]//2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            ret[c_y+(y-_c_y),c_x+(x-_c_x)] = img[y,x]
    return ret

# %%

img =  Padding((512,512),img)
mask =  Padding((512,512),GenerateMask(img)*np.uint8(255))

# %%
cv2.imwrite(image_save,img)
cv2.imwrite(mask_save,mask)

# %%
from inference import Create3DModel

Create3DModel(image_save,mask_save,model_save)

# %%



