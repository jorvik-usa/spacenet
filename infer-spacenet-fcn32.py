import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe

ifile='3band_013022232022_Public_img6684.png'
im = Image.open('../spacenet/processedData/JPEGImages/'+ifile)
seg = Image.open('../spacenet/processedData/SegmentationClass/'+ifile)

in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((70.40782668, 77.32761001, 55.56687855))
in_ = in_.transpose((2,0,1))

# load net
caffe.set_mode_gpu()
net = caffe.Net('deploy.prototxt', 'snapshot/train_iter_2500.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score_SN'].data[0]


fig=plt.figure()
fig.suptitle(ifile, fontsize=16)
ax=plt.subplot(1,3,1)
ax.set_title("Image")
ax.imshow(np.array(im))
ax.axis('off')
ax=plt.subplot(1,3,2)
ax.set_title("Inferred")
ax.imshow(out.argmax(axis=0),cmap='gray')
ax.axis('off')
ax=plt.subplot(1,3,3)
ax.set_title("Ground truth")
ax.imshow(np.array(seg),cmap='gray')
ax.axis('off')
plt.show()
