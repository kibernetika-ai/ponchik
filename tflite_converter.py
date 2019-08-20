import argparse
import glob
import os
import itertools

import cv2
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph',
    )
    parser.add_argument(
        '--saved-model'
    )
    parser.add_argument(
        '--data-dir'
    )
    parser.add_argument(
        '--output',
        default='model.tflite'
    )
    parser.add_argument(
        '--limit-steps',
        type=int,
        default=0,
    )
    return parser.parse_args()


def normalize(img):
    normalized = img / 127.5 - 1.0
    return normalized.astype(np.float32)


def main():
    args = parse_args()

    def representative_dataset_gen():
        dataset_dir = args.data_dir.rstrip('/')
        jpg_iter = glob.iglob(dataset_dir + '/**/*.jpg')
        jpeg_iter = glob.iglob(dataset_dir + '/**/*.jpeg')
        png_iter = glob.iglob(dataset_dir + '/**/*.png')
        iterator = itertools.chain(jpg_iter, jpeg_iter, png_iter)
        i = 0
        for path in iterator:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            i += 1
            if args.limit_steps != 0 and i >= args.limit_steps:
                break

            yield [np.expand_dims(normalize(img), axis=0)]

    if args.graph:
        converter = tf.lite.TFLiteConverter.from_frozen_graph(
            args.graph,
            input_arrays=['input'],
            output_arrays=['embeddings'],
            input_shapes={'input': [1, 160, 160, 3]}
        )
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)

    converter.allow_custom_ops = True
    converter.target_ops.add(tf.lite.OpsSet.SELECT_TF_OPS)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()

    dirname = os.path.dirname(args.output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(args.output, 'wb') as f:
        f.write(tflite_quant_model)
    # __import__('ipdb').set_trace()


if __name__ == '__main__':
    main()
