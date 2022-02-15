import tensorflow as tf
import cv2
import numpy as np
import os
import argparse
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from DepthEstimation import network
from tensorflow.python.util import deprecation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()
#Variables
cpu_only=True
test_imgs="Input"
ckpt_pth="DepthEstimation/ckpt/pydnet"
original_size=True
dest_pth="Output"

if cpu_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def create_dir(d):
    """ Create a directory if it does not exist
    Args:
        d: directory to create
    """
    if not os.path.exists(d):
        os.makedirs(d)


def main():
    network_params = {"height": 320, "width": 640, "is_training": False}

    if os.path.isfile(test_imgs):
        img_list = [test_imgs]
    elif os.path.isdir(test_imgs):
        img_list = glob.glob(os.path.join(test_imgs, "*.{}".format("jpg")))
        img_list = sorted(img_list)
        if len(img_list) == 0:
            raise ValueError("No {} images found in folder {}".format(".jpg", test_imgs))
        print("=> found {} images".format(len(img_list)))
    else:
        raise Exception("No image nor folder provided")

    model = network.Pydnet(network_params)
    tensor_image = tf.compat.v1.placeholder(tf.float32, shape=(320, 640, 3))
    batch_img = tf.expand_dims(tensor_image, 0)
    tensor_depth = model.forward(batch_img)
    tensor_depth = tf.nn.relu(tensor_depth)

    # restore graph
    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, ckpt_pth)

    # run graph
    for i in tqdm(range(len(img_list))):

        # preparing image
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img = cv2.resize(img, (640, 320))
        img = img / 255.0

        # inference
        depth = sess.run(tensor_depth, feed_dict={tensor_image: img})
        depth = np.squeeze(depth)
        min_depth = depth.min()
        max_depth = depth.max()
        depth = (depth - min_depth) / (max_depth - min_depth)
        depth *= 255.0

        # preparing final depth
        if original_size:
            depth = cv2.resize(depth, (w, h))
        name = os.path.basename(img_list[i]).split(".")[0]
        dest = dest_pth
        np_file=dest+"/"+name
        create_dir(dest)
        dest = os.path.join(dest, name + ".jpg")
        np.save(np_file,depth)
        plt.imsave(dest, depth, cmap="magma")
