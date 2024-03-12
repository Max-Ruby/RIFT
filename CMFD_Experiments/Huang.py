# This runs the method by
# Yanping Huang et. al. developed in
# "Improved DCT-based detection of copy-move forgery in images."

# Errors can be addressed to the email on the resume.

import cv2
import numpy as np

# Let's grab the parameters from the paper
# They're in section 3.1
B = 8
T = 35
N_f = 3
N_d = 16
p = 0.25
q = 4
s_thresh = 4
t_thresh = 0.0625

# First read in the images

image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)

# Second run Huang

# 1) Convert to Grayscale...was done on read.

# 1.5) Pad image to a multiple of BxB so we can get blocks. No padding is specified in the paper.
# This is a DCT-based method, so border_wrap is best?

x_dim = image.shape[0]
y_dim = image.shape[1]
x_pad = x_dim - B*(x_dim//B)
y_pad = y_dim - B*(y_dim//B)

image = cv2.copyMakeBorder(image,x_pad//2,x_pad-x_pad//2,y_pad//2,y_pad-y_pad//2,cv2.BORDER_WRAP)

# 1.75) Rescale image to live in [0,1]
image = image - np.min(np.min(image))
image = image/(np.max(np.max(image)))

# 2) A fixed BxB window is slid one pixel along the upper left corner to the bottom right, dividing the image into a number of blocks
# We only do this mentally.

# 3) Apply DCT to each block, and reshape the BxB matrix into a row vector using a "zig-zag scan".
# I think the zig-zag scan is arbitrary here, and it's not really clear what they mean by this.
# I'll assume this translates into "raster" ordering.
# Moreover, I think reshaping it into a row is arbitrary given how they're handling it.

DCT_storage = np.zeros([image.shape[0]//B,image.shape[1]//B,int(p*B*B)+2])

for i in range(0,image.shape[0]//B):
    for j in range(0,image.shape[1]//B):
        DCT_block= cv2.dct(image[B * i:B * (i + 1), B * j:B * (j + 1)])
        print("-")
        print(np.min(np.min(np.fabs(DCT_block))))
        DCT_block_flat = np.fabs(DCT_block.flatten())
        DCT_block_flat.sort(axis=0)
        key = int((1-p)*B*B)
        DCT_threshold = DCT_block_flat[key]
        DCT_block_clobber = np.clip(DCT_block,-DCT_threshold,DCT_threshold)
        DCT_block = DCT_block - DCT_block_clobber

# 4) Sort lexicographically to form a new matrix, A.
# Make sure to keep the i,j key attached to the rows.

# 5) For each row in A, test the neighboring rows for similarity.
# It looks like they do this with L_1 norm.
#

# 6) Compute distance between blocks.

# 7) If distance exceeds a threshold, compute/normalize shift vector. Then add 1 to the shift vector's frequency
# I guess this "shift" vector's "frequency" isn't the DCT's frequency - it must be "similarity block count."

# 8) If there's blocks over the threshold, T, then mark the corresponding blocks. Those are the suspicious ones.

# Third spit out metrics

