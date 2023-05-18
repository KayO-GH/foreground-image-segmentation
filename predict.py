import argparse
import tensorflow as tf
import os
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt

BATCH_SIZE = 8
output_folder = ""

def load_data(path, multiple):
    if multiple:
        images = sorted(glob(os.path.join('./', path+'/*')))
    else:
        images = sorted(glob(os.path.join('./', path)))
    return images

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (128, 128))
    x = x/255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (128, 128))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([128, 128, 3])
    y.set_shape([128, 128, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    return dataset


def display(display_item, size, name):
    # Resize the image to original, set to b/w, set the name and save to ouput folder
    pred_mask = tf.keras.preprocessing.image.array_to_img(display_item).resize(size=size)
    output_image = cv2.cvtColor(np.array(pred_mask), cv2.COLOR_RGB2BGR) #convert to b/w
    output_location = './'+output_folder+('/' if not output_folder.endswith('/') else '')+name.split('/')[-1]
    output_location = output_location.replace('.jpg', '_no_seed.png')
    cv2.imwrite(output_location, output_image)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def make_prediction(dataset, model, size, name, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display(create_mask(pred_mask[0]), size, name)
   
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get masks for images.')
    parser.add_argument('input_path')
    parser.add_argument('seed_path')
    parser.add_argument('output_path')
    parser.add_argument('--multiple', '-m', dest="multiple", action='store_true', help='get masks for multiple images (default: get mask for single image)')
    # We ignore the seed files 

    args = parser.parse_args()

    input_path = args.input_path
    output_folder = args.output_path

    # Load saved model
    model =  tf.keras.models.load_model("./segment_model")

    print("model loaded")
    images = load_data(args.input_path, args.multiple)

    for image in images:
        # get and resize
        original_img = cv2.imread(image)
        h, w, _ = original_img.shape

        test = tf_dataset([image], [image], batch=BATCH_SIZE)
        test_dataset = test.batch(BATCH_SIZE)

        make_prediction(test_dataset, model, (w,h), image)


        