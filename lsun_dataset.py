import tensorflow as tf
import numpy as np
import cv2
import os
import sys
from scipy import io

NUM_OF_CLASSESS = 11
NUM_OF_POINTS = (5 * 5 * 3 * 3)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def build_data(path):
    sets = [('training', 2), ('validation', 1)]

    for set in sets:
        tfrecords_filename = '%s/lsun_%s.tfrecords' % (path, set[0])
        if os.path.exists(tfrecords_filename):
            print('TFRecord file already exist: %s' % tfrecords_filename)
            continue
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        datas = io.loadmat('%s/%s.mat' % (path, set[0]))[set[0]][0]

        for data in datas:
            for i in xrange(set[1]):
                filename = '%s.jpg' % data[0][0]
                with tf.gfile.FastGFile('%s/images/%s' % (path, filename), 'rb') as f:
                    image = f.read()
                    if image[0] != '\xff' or image[1] != '\xd8':
                        print('Invalid JPEG file: %s' % filename)
                        continue
                room_type = np.zeros(NUM_OF_CLASSESS)
                room_type[data[2][0][0]] = 1
                width = data[4][0][1]
                height = data[4][0][0]
                num_points = len(data[3])
                points = np.copy(data[3])
                for j in xrange(num_points):
                    points[j][0] /= width
                    points[j][1] /= height
                points = np.reshape(points, (num_points * 2))
                #print(room_type, num_points, width, height)

                if i == 1:
                    image = cv2.imread('%s/images/%s' % (path, filename))
                    image = cv2.flip(image, 1)
                    image = cv2.imencode('.jpg', image)[1].tostring()
                    for j in xrange(0, num_points * 2, 2):
                        points[j] = 1 - points[j]

                points_map = np.zeros((5, 5, 3, 3))
                shape = points_map.shape
                span = 1. / shape[0]
                for j in xrange(0, num_points * 2, 2):
                    cell_x = abs(points[j] // span)
                    offset_x = (points[j] % span) / span
                    if int(cell_x) == shape[0]:
                        cell_x -= 1.
                        offset_x = 1.

                    cell_y = abs(points[j + 1] // span)
                    offset_y = (points[j + 1] % span) / span
                    if int(cell_y) == shape[0]:
                        cell_y -= 1.
                        offset_y = 1.

                    for c in xrange(shape[2]):
                        if points_map[int(cell_y)][int(cell_x)][c][0] == 0.:
                            points_map[int(cell_y)][int(cell_x)][c][0] = 1.
                            points_map[int(cell_y)][int(cell_x)][c][1] = offset_y
                            points_map[int(cell_y)][int(cell_x)][c][2] = offset_x
                            break

                label = np.array(room_type)
                label = np.append(label, points_map)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'filename': _bytes_feature(bytes(filename)),
                    'image': _bytes_feature(image),
                    'width': _int64_feature(width),
                    'height': _int64_feature(height),
                    'num_points': _int64_feature(num_points),
                    'label': _float_feature(label),
                }))
                writer.write(example.SerializeToString())
        writer.close()

def read_data(path, set, image_size, batch_size):
    filename_queue = tf.train.string_input_producer(['%s/lsun_%s.tfrecords' % (path, set)])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'num_points': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([NUM_OF_CLASSESS + NUM_OF_POINTS], tf.float32),
        })

    filename = features['filename']
    image = tf.image.decode_jpeg(features['image'])
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    num_points = tf.cast(features['num_points'], tf.int32)
    label = features['label']

    if set == 'training':
        image = tf.image.resize_images(tf.reshape(image, [height, width, 3]), [image_size + 8, image_size + 8])
        image = tf.random_crop(image, [image_size, image_size, 3])

        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

        filename_batch, image_batch, width_batch, height_batch, num_points_batch, label_batch = \
            tf.train.shuffle_batch([filename, image, width, height, num_points, label],
                batch_size=batch_size,
                capacity=10000,
                num_threads=4,
                min_after_dequeue=4000)
    else:
        image = tf.image.resize_images(tf.reshape(image, [height, width, 3]), [image_size, image_size])

        filename_batch, image_batch, width_batch, height_batch, num_points_batch, label_batch = \
            tf.train.batch([filename, image, width, height, num_points, label],
               batch_size=batch_size,
               capacity=1000,
               num_threads=4)

    return filename_batch, image_batch, width_batch, height_batch, num_points_batch, label_batch

def test(path):
    IMAGE_SIZE = 224
    BATCH_SIZE = 32

    filename_op, image_op, width_op, height_op, num_points_op, label_op = \
        read_data(path, 'training', IMAGE_SIZE, BATCH_SIZE)

    with tf.Session()  as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        filename, image, width, height, num_points, label = \
            sess.run([filename_op, image_op, width_op, height_op, num_points_op, label_op])

        for i in xrange(BATCH_SIZE):
            room_type = np.argmax(label[i][:NUM_OF_CLASSESS])
            points = label[i][NUM_OF_CLASSESS:]
            print(i, filename[i], width[i], height[i], room_type, num_points[i])
            img = image[i].astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            points = np.reshape(points, (5, 5, 3, 3))
            shape = points.shape
            span = 1. / shape[0]
            for y in xrange(shape[0]):
                for x in xrange(shape[1]):
                    for c in xrange(shape[2]):
                        if points[y][x][c][0] == 1.:
                            point_y = int((y * span + points[y][x][c][1] * span) * IMAGE_SIZE)
                            point_x = int((x * span + points[y][x][c][2] * span) * IMAGE_SIZE)
                            cv2.circle(img, (point_x, point_y), 5, (0, 0, 255), -1)

            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if key == 1048603:
                break

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    mode = sys.argv[1]
    path = sys.argv[2]
    if mode == 'build':
        build_data(path)
    elif mode == 'test':
        test(path)