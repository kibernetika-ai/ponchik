# urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
# window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));


# import os
# import requests
#
#
# urls = open('/Users/vadim/Downloads/lopez.csv').read().strip().split("\n")
# for url in urls:
#     try:
#         request = requests.get(url, timeout=4, stream=True)
#     except Exception as e:
#         print("Download %s error: %s" % (url, e))
#         continue
#     base_name = os.path.basename(url).split('?')[0]
#     if base_name == '':
#         continue
#     save_to = os.path.join('./data/to_clusterize', base_name)
#     with open(save_to, 'wb') as fh:
#         for chunk in request.iter_content(1024 * 1024):
#             fh.write(chunk)
#     print(url)


# import cv2
# import os
# import numpy as np
# from app.dataset.aligner import Aligner
# from app.tools import images
#
#
# src_path = './data/to_clusterize'
# dest_path = './data/to_clusterize_aligned'
# image_paths = [os.path.join(src_path, f) for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
# aligner = Aligner()
# aligner._load_driver()
# for image_path in image_paths:
#     try:
#         img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
#     except:
#         print('WARNING: Unable to read image "%s"' % image_path)
#         continue
#     if len(img.shape) <= 2:
#         print('WARNING: Unable to align "%s", shape %s' % (image_path, img.shape))
#         continue
#     bounding_boxes = aligner._get_boxes(image_path, img)
#     imgs = images.get_images(
#         img,
#         bounding_boxes,
#         face_crop_size=aligner.image_size,
#         face_crop_margin=aligner.margin,
#         do_prewhiten=False,
#     )
#     output_filename = os.path.join(dest_path, os.path.splitext(os.path.split(image_path)[1])[0] + '.png')
#     for i, cropped in enumerate(imgs):
#         bb = bounding_boxes[i]
#         filename_base, file_extension = os.path.splitext(output_filename)
#         output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
#         cv2.imwrite(output_filename_n, cropped)
#     print(image_path)


import os
from app.recognize.classifiers import Classifiers
from sklearn.cluster import DBSCAN
import numpy as np
import shutil

faces_path = './data/to_clusterize_aligned'
done_path = './data/to_clusterize_done'
faces_paths = [
    os.path.join(faces_path, f) for f in os.listdir(faces_path) if os.path.isfile(os.path.join(faces_path, f))]

# print(faces_paths)

clf = Classifiers(aug_noise=0, aug_flip=False)
clf.load_model()

# for face_path in faces_paths:
imgs = clf.load_data(faces_paths, [1] * len(faces_paths))
embs = clf.embeddings(imgs)
print(embs.shape)

# clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"])
print('start')
clt = DBSCAN(eps=0.7)
clt.fit(embs)
print('done')

labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

print(clt.labels_)

shutil.rmtree(done_path, ignore_errors=True)
os.makedirs(done_path)

for i, src_path in enumerate(faces_paths):
    desc_path = os.path.join(done_path, str(clt.labels_[i]), os.path.basename(src_path))
    # print('%s -> %s' % (src_path, desc_path))
    if not os.path.isdir(os.path.dirname(desc_path)):
        os.makedirs(os.path.dirname(desc_path))
    shutil.copyfile(src_path, desc_path)

# for labelID in labelIDs:
#     # copyfile()
#     # find all indexes into the `data` array that belong to the
#     # current label ID, then randomly sample a maximum of 25 indexes
#     # from the set
#     print("[INFO] faces for face ID: {}".format(labelID))
#     # idxs = np.where(clt.labels_ == labelID)[0]
#     # idxs = np.random.choice(idxs, size=min(25, len(idxs)),
#     #                         replace=False)
#
#     # initialize the list of faces to include in the montage
#     faces = []
