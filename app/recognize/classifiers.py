import hashlib
import json
import math
import os
import pickle
import shutil
import time

import numpy as np
from ml_serving.drivers import driver
from sklearn import svm, neighbors

from app.recognize import defaults
from app.tools import utils
from app.tools import dataset
from app.tools import images

META_FILENAME = 'meta.json'
print_fun = utils.print_fun


def add_classifier_args(parser):
    parser.add_argument(
        '--aligned_dir',
        type=str,
        help='Directory with aligned face thumbnails.',
        default=defaults.ALIGNED_DIR,
    )
    parser.add_argument(
        '--image_size',
        type=int,
        help='Image size (height, width) in pixels.',
        default=defaults.IMAGE_SIZE,
    )
    parser.add_argument(
        '--margin',
        type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.',
        default=defaults.IMAGE_MARGIN,
    )
    parser.add_argument(
        '--complementary_train',
        help='Existing embeddings supplements with new ones from aligned dir for training.',
        action='store_true',
    )
    parser.add_argument(
        '--aug_flip',
        type=utils.boolean_string,
        default=defaults.AUG_FLIP,
        help='Add horizontal flip to images.',
    )
    parser.add_argument(
        '--aug_noise',
        type=int,
        default=defaults.AUG_NOISE,
        help='Noise count for each image.',
    )
    parser.add_argument(
        '--aug_blur',
        type=utils.boolean_string,
        default=True,
        help='Add gaussian blur to images.',
    )
    parser.add_argument(
        '--aug_upscale',
        type=utils.boolean_string,
        default=defaults.AUG_UPSCALE,
        help='Add upscale to images from 30x30 size.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Number of images to process in a batch.',
        default=defaults.BATCH_SIZE,
    )
    parser.add_argument(
        '--with_svm',
        action='store_true',
        help='Use SVM classifier for training and recognition.',
    )
    parser.add_argument(
        '--store_embeddings',
        action='store_true',
        help='Save embeddings to model for debug.',
    )


def classifiers_args(args):
    return Classifiers(
        aligned_dir=args.aligned_dir,
        complementary_train=args.complementary_train or args.complementary,
        normalization=args.normalization,
        classifiers_dir=args.classifiers_dir,
        model_path=args.model_path,
        aug_flip=args.aug_flip,
        aug_noise=args.aug_noise,
        aug_blur=args.aug_blur,
        aug_upscale=args.aug_upscale,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        with_svm=args.with_svm,
        store_embeddings=args.store_embeddings,
        best_threshold=args.best_threshold,
    )


class Classifiers:
    def __init__(
            self,
            aligned_dir=defaults.ALIGNED_DIR,
            complementary_train=False,
            normalization=defaults.NORMALIZATION,
            classifiers_dir=defaults.CLASSIFIERS_DIR,
            model_path=defaults.MODEL_PATH,
            aug_flip=defaults.AUG_FLIP,
            aug_noise=defaults.AUG_NOISE,
            aug_blur=False,
            aug_upscale=defaults.AUG_UPSCALE,
            image_size=defaults.IMAGE_SIZE,
            batch_size=defaults.BATCH_SIZE,
            device=defaults.DEVICE,
            with_svm=False,
            store_embeddings=False,
            best_threshold=False,
    ):
        self.algorithms = ["kNN"]
        if with_svm:
            self.algorithms.append("SVM")
        self.driver_name = "openvino"
        self.aligned_dir = aligned_dir
        self.complementary_train = complementary_train
        self.normalization = normalization
        self.classifiers_dir = classifiers_dir
        self.model_path = model_path
        self.aug_flip = aug_flip
        self.aug_noise = aug_noise
        self.aug_blur = aug_blur
        self.aug_upscale = aug_upscale
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device
        self.loading_image_total_time = 0
        self.loading_images = 0
        self.serving = None
        self.store_embeddings = store_embeddings
        self.find_best_threshold = best_threshold

    def train(self):

        loaded_dataset = dataset.get_dataset(self.aligned_dir)

        # Check that there are at least one training image per class
        for cls in loaded_dataset:
            if len(cls.image_paths) == 0:
                print_fun('WARNING: %s: There are no aligned images in this class.' % cls)

        paths, labels = dataset.split_to_paths_and_labels(loaded_dataset)

        print_fun('Number of classes: %d' % len(loaded_dataset))
        print_fun('Number of images: %d' % len(paths))

        # Load the model
        print_fun('Loading feature extraction model')

        # Load and instantinate driver
        self._load_model()

        # Run forward pass to calculate embeddings
        print_fun('Calculating features for images')

        emb_args = {
            'model': self.model_path,
            'noise': self.aug_noise,
            'flip': self.aug_flip,
            'blur': self.aug_blur,
            'upscale': self.aug_upscale,
            'image_size': self.image_size,
            'normalization': self.normalization,
        }

        stored_embeddings = {}
        emb_fn = os.path.join(
            self.aligned_dir,
            "embeddings-%s.pkl" % hashlib.md5(json.dumps(emb_args, sort_keys=True).encode()).hexdigest(),
        )
        if os.path.isfile(emb_fn):
            print_fun("Found stored embeddings data, loading...")
            with open(emb_fn, 'rb') as embeddings_file:
                stored_embeddings = pickle.load(embeddings_file)

        if not self.complementary_train:
            deleted_embs, deleted_classes = 0, 0
            for stored_class in list(stored_embeddings):
                for stored_image in list(stored_embeddings[stored_class]):
                    if not os.path.isfile(stored_image):
                        del stored_embeddings[stored_class][stored_image]
                        deleted_embs += 1
                if len(stored_embeddings[stored_class]) == 0:
                    del stored_embeddings[stored_class]
                    deleted_classes += 1
            if deleted_embs > 0 or deleted_classes > 0:
                print_fun("Deleted not existing in aligned data %d embeddings and %d classes"
                          % (deleted_embs, deleted_classes))

        total_time = 0.

        nrof_images = len(paths)

        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        # for each file: (original + flipped) * noise
        # embeddings_size = nrof_images * (1 + self.aug_noise) * (2 if self.aug_flip else 1)
        embeddings_size = self.aug_images_count(nrof_images)

        emb_array = np.zeros((embeddings_size, 512))
        fit_labels = []

        emb_index = 0
        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            labels_batch = labels[start_index:end_index]

            # has_not_stored_embeddings = False
            paths_batch_load, labels_batch_load = [], []

            for j in range(end_index - start_index):
                # print_fun(os.path.split(paths_batch[j]))
                cls_name = loaded_dataset[labels_batch[j]].name
                cached = True
                if cls_name not in stored_embeddings or paths_batch[j] not in stored_embeddings[cls_name]:
                    # has_not_stored_embeddings = True
                    cached = False
                    paths_batch_load.append(paths_batch[j])
                    labels_batch_load.append(labels_batch[j])
                else:
                    embeddings = stored_embeddings[cls_name][paths_batch[j]]
                    emb_array[emb_index:emb_index + len(embeddings), :] = stored_embeddings[cls_name][paths_batch[j]]
                    fit_labels.extend([labels_batch[j]] * len(embeddings))
                    emb_index += len(embeddings)

                if not cached:
                    print_fun('Batch {} <-> {} {} {}'.format(
                        paths_batch[j], labels_batch[j], cls_name, "cached" if cached else "",
                    ))

            if len(paths_batch_load) == 0:
                continue

            t = time.time()
            imgs = self.load_data(paths_batch_load, labels_batch_load)
            print_fun("Load & Augmentation: %.3fms" % ((time.time() - t) * 1000))

            t = time.time()
            emb_outputs = self._predict(imgs)
            print_fun("Inference: %.3fms" % ((time.time() - t) * 1000))

            for n, e in enumerate(emb_outputs):
                cls_name = loaded_dataset[labels_batch_load[n]].name
                if cls_name not in stored_embeddings:
                    stored_embeddings[cls_name] = {}
                path = paths_batch_load[n]
                if path not in stored_embeddings[cls_name]:
                    stored_embeddings[cls_name][path] = []
                stored_embeddings[cls_name][path].append(e)

            emb_array[emb_index:emb_index + len(imgs), :] = emb_outputs
            fit_labels.extend(labels_batch_load)

            emb_index += len(imgs)

        # average_time = total_time / embeddings_size * 1000
        # print_fun('Average time: %.3fms' % average_time)

        if self.loading_images != 0:
            print_fun("Load image file requests count: %d" % self.loading_images)
            print_fun("Load images total time: %.3fs" % self.loading_image_total_time)
            print_fun("Load images average time: %.3fms" % ((self.loading_image_total_time / self.loading_images) * 1000))

        classifiers_dir = os.path.expanduser(self.classifiers_dir)

        # Save embeddings
        with open(emb_fn, 'wb') as emb_f:
            pickle.dump(stored_embeddings, emb_f, protocol=2)

        # Clear (or create) classifiers directory
        if os.path.exists(classifiers_dir):
            shutil.rmtree(classifiers_dir, ignore_errors=True)
        os.makedirs(classifiers_dir)

        # Save embeddings to model
        if self.store_embeddings:
            with open(embeddings_filename(classifiers_dir), 'wb') as emb_f:
                pickle.dump(stored_embeddings, emb_f, protocol=2)

        # Create a list of class names
        dataset_class_names = [cls.name for cls in loaded_dataset]
        class_names = [cls.replace('_', ' ') for cls in dataset_class_names]

        class_stats = [{} for _ in range(len(dataset_class_names))]
        for cls in stored_embeddings:
            class_stats[dataset_class_names.index(cls)] = {
                'images': len(stored_embeddings[cls]),
                'embeddings': sum(len(e) for e in stored_embeddings[cls].values()),
            }

        # Train classifiers
        for algorithm in self.algorithms:

            print_fun('Classifier algorithm %s' % algorithm)
            # update_data({'classifier_algorithm': args.algorithm}, use_mlboard, mlboard)
            if algorithm == 'SVM':
                clf = svm.SVC(kernel='linear', probability=True)
            elif algorithm == 'kNN':
                # n_neighbors = int(round(np.sqrt(len(emb_array))))
                n_neighbors = 1
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
            else:
                raise RuntimeError("Classifier algorithm %s not supported" % algorithm)

            if len(emb_array) == 0:
                raise RuntimeError("Empty fit data")
            clf.fit(emb_array, fit_labels)

            # Saving classifier model
            clf_fn = classifier_filename(classifiers_dir, algorithm)
            with open(clf_fn, 'wb') as outfile:
                pickle.dump((clf, class_names, class_stats), outfile, protocol=2)
            print_fun('Saved classifier model to file "%s"' % clf_fn)

            # update_data({'average_time_%s': '%.3fms' % average_time}, use_mlboard, mlboard)

        previews_dir = self.store_previews()
        print_fun('Saved class previews to dir "%s"' % previews_dir)

        meta = dataset.get_meta(loaded_dataset)
        meta_file = os.path.join(classifiers_dir, META_FILENAME)
        if len(meta) > 0:
            with open(meta_file, 'w') as mw:
                json.dump(meta, mw)
                print_fun('Saved metadata to "%s"' % meta_file)
        else:
            shutil.rmtree(meta_file, ignore_errors=True)
            print_fun('No metadata saved')

        if self.find_best_threshold:
            self.store_best_threshold(stored_embeddings, classifiers_dir)

        print_fun("Training complete: created %d classifiers with %d embeddings (%d images) in %d classes" %
                  (len(self.algorithms), len(emb_array), nrof_images, len(set(fit_labels))))

    def store_best_threshold(self, stored_embeddings, dir):
        thresholds = np.arange(0, 1, 0.0025)
        embedding_size = 512
        size = 0
        for i, cls in enumerate(stored_embeddings):
            for embs in stored_embeddings[cls].values():
                size += len(embs)

        plain_embeddings = np.zeros([size, embedding_size])
        class_index = []
        emb_i = 0
        for i, cls in enumerate(stored_embeddings):
            for embs in stored_embeddings[cls].values():
                for emb in embs:
                    plain_embeddings[emb_i] = emb
                    emb_i += 1
                    class_index.append(cls)

        embeddings = (plain_embeddings + 1.) / 2.
        kd_tree = neighbors.KDTree(embeddings, metric='euclidean')

        dists, recognized = np.zeros([size], dtype=np.float32), np.zeros([size], np.bool)
        for i, emb in enumerate(embeddings):
            dist, idx = kd_tree.query(emb.reshape([1, 512]), k=2)

            dist = dist[0][np.argmax(dist)]
            idx = idx[0][np.argmax(dist)]
            detected_class = class_index[idx]

            detected = detected_class == class_index[i]
            recognized[i] = detected
            dists[i] = dist

        # __import__('ipdb').set_trace()
        best_threshold = 0
        max_detect = 0
        if recognized.all():
            best_threshold = np.max(dists)
        else:
            for threshold in thresholds:
                detected = len(dists[recognized & (dists < threshold)])
                if detected > max_detect:
                    max_detect = detected
                    best_threshold = threshold

        threshold_file = os.path.join(dir, 'threshold.txt')
        best_threshold = best_threshold * 1.1

        print_fun('=' * 50)
        print_fun('Found best threshold = %s' % best_threshold)
        print_fun('Written to %s.' % threshold_file)
        print_fun('=' * 50)

        with open(threshold_file, 'w') as f:
            f.write(str(best_threshold) + '\n')

    def _load_model(self):
        if self.serving is None:
            drv = driver.load_driver(self.driver_name)
            self.serving = drv()
            self.serving.load_model(
                self.model_path,
                inputs='input:0,phase_train:0',
                outputs='embeddings:0',
                device=self.device,
                flexible_batch_size=True,
            )

    def _predict(self, imgs):
        if self.serving.driver_name == 'tensorflow':
            feed_dict = {'input:0': imgs, 'phase_train:0': False}
        elif self.serving.driver_name == 'openvino':
            input_name = list(self.serving.inputs.keys())[0]
            # Transpose image for channel first format
            imgs = imgs.transpose([0, 3, 1, 2])
            feed_dict = {input_name: imgs}
        else:
            raise RuntimeError('Driver %s currently not supported' % self.serving.driver_name)
        outputs = self.serving.predict(feed_dict)
        return list(outputs.values())[0]

    def load_data(self, paths_batch, labels):
        if len(paths_batch) != len(labels):
            raise RuntimeError("load_data: len(paths_batch) = %d != len(labels) = %d", len(paths_batch), len(labels))

        init_batch_len = len(paths_batch)

        t = time.time()
        imgs = dataset.load_data(paths_batch, self.image_size, fixed_normalization=self.normalization)
        self.loading_images += len(paths_batch)
        self.loading_image_total_time += (time.time() - t)

        imgs = self.apply_augmentation(paths_batch, labels, imgs)

        batch_log = ' ... %d images' % len(imgs)
        # if self.aug_noise > 0 or self.aug_flip:
        # batch_log_details = ['%d original' % init_batch_len]
        batch_log_aug = ['%d original' % init_batch_len]
        if self.aug_noise > 0:
            batch_log_aug.append('{} noise'.format(self.aug_noise * init_batch_len))
            # batch_log_details.append('%d noise' % (init_batch_len * self.aug_noise))
        if self.aug_blur:
            batch_log_aug.append('{} blur'.format(init_batch_len))
        if self.aug_upscale:
            batch_log_aug.append('{} upscale'.format(init_batch_len))
        if len(batch_log_aug) > 1 or self.aug_flip:
            batch_log_aug = ', '.join(batch_log_aug)
            if self.aug_flip:
                batch_log_aug = '({}) * 2(flip each)'.format(batch_log_aug)
                # batch_log_details.append('%d flip' % init_batch_len)
            # if self.aug_noise > 0 and self.aug_flip:
            #     batch_log_details.append('%d noise+flip' % (init_batch_len * self.aug_noise))
            batch_log = '%s (%s)' % (batch_log, batch_log_aug)
        print_fun(batch_log)

        return imgs

    def apply_augmentation(self, paths_batch, labels, imgs, already_flipped=False):
        imgs_size = len(imgs)
        for k in range(imgs_size):
            img = imgs[k]
            if self.aug_noise > 0:
                for i in range(self.aug_noise):
                    # print_fun('Applying noise to image {}, #{}'.format(paths_batch[k], i + 1))
                    noised = images.random_noise(img)
                    imgs = np.concatenate((imgs, noised.reshape(1, *noised.shape)))
                    labels.append(labels[k])
                    paths_batch.append(paths_batch[k])
            if self.aug_blur:
                blured = images.blur(img)
                imgs = np.concatenate((imgs, blured.reshape(1, *blured.shape)))
                labels.append(labels[k])
                paths_batch.append(paths_batch[k])
            if self.aug_upscale:
                upscaled = images.upscale(img)
                imgs = np.concatenate((imgs, upscaled.reshape(1, *upscaled.shape)))
                labels.append(labels[k])
                paths_batch.append(paths_batch[k])
            if self.aug_flip and not already_flipped:
                flipped = images.horizontal_flip(img)
                flipped_imgs = [flipped]
                flipped_labels = [labels[k]]
                flipped_paths_batch = [paths_batch[k]]
                flipped_imgs_aug = self.apply_augmentation(
                    flipped_paths_batch, flipped_labels, flipped_imgs, already_flipped=True)
                for flipped_img_aug in flipped_imgs_aug:
                    imgs = np.concatenate((imgs, flipped_img_aug.reshape(1, *flipped_img_aug.shape)))
                labels.extend(flipped_labels)
                paths_batch.extend(flipped_paths_batch)
        return imgs

    def aug_images_count(self, count):
        m = 1  # original image
        if self.aug_noise > 0:
            m += self.aug_noise
        if self.aug_blur:
            m += 1
        if self.aug_upscale:
            m += 1
        if self.aug_flip:
            m = m * 2
        return count * m

    def store_previews(self):
        previews_dir = os.path.join(self.classifiers_dir, "previews")
        if os.path.exists(previews_dir):
            shutil.rmtree(previews_dir, ignore_errors=True)
        os.makedirs(previews_dir)
        if os.path.isdir(self.aligned_dir):
            for d in os.listdir(self.aligned_dir):
                dd = os.path.join(self.aligned_dir, d)
                if os.path.isdir(dd):
                    for f in os.listdir(dd):
                        ff = os.path.join(dd, f)
                        if os.path.isfile(ff):
                            _, ext = os.path.splitext(ff)
                            if ext.lower() == ".png":
                                shutil.copyfile(ff, os.path.join(previews_dir, "%s.png" % d))
                                break
        return previews_dir


def classifier_filename(classifiers_dir, algorithm):
    return os.path.join(classifiers_dir, "classifier-%s.pkl" % algorithm.lower())


def embeddings_filename(classifiers_dir):
    return os.path.join(classifiers_dir, "embeddings.pkl")
