from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import os
import random
from six.moves import xrange
from sklearn.metrics import mean_squared_error

import TensorflowUtils as utils
import lsun_dataset

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '32', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '100000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'Data_zoo/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.0001', 'Learning rate for Momentum Optimizer')
tf.flags.DEFINE_string('model_dir', 'Model_zoo/', 'Path to vgg model mat')
tf.flags.DEFINE_bool('debug', 'True', 'Debug mode: True/ False')
tf.flags.DEFINE_string('mode', 'train', 'Mode train/ val/ test')
tf.flags.DEFINE_string('images_dir', '', 'path to test images')
tf.flags.DEFINE_float('thresh', '0.5', 'threshold for points')

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = 11
NUM_OF_CELL = 5
NUM_OF_POINTS = (NUM_OF_CELL * NUM_OF_CELL * 3 * 3)
IMAGE_SIZE = 224

def vgg_net(weights, image, is_train):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name='%s_w' % name)
            current = utils.conv2d_basic(current, kernels, None)
            current = utils.batch_norm(current, kernels.get_shape()[3], is_train, '%s_bn' % name)
        elif kind == 'relu':
            current = tf.nn.relu(current, name='%s' % name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def inference(image, keep_prob, is_train):
    '''
    Room layout estimation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    '''
    print('setting up vgg initialized conv layers ...')
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    weights = np.squeeze(model_data['layers'])
    processed_image = tf.div(tf.subtract(image, 128), 128)

    with tf.variable_scope('inference'):
        image_net = vgg_net(weights, processed_image, is_train)
        conv_final_layer = image_net['relu5_4']
        pool5 = utils.max_pool_2x2(conv_final_layer)

        dropout = tf.nn.dropout(pool5, keep_prob, name='dropout')
        conv6 = tf.layers.conv2d(dropout, NUM_OF_CLASSESS + NUM_OF_POINTS, [1, 1], name='conv6')
        size = conv6.get_shape().as_list()[1]
        avg_pool = tf.layers.average_pooling2d(conv6, [size, size], [size, size], name='avg_pool')
        avg_pool = tf.reshape(avg_pool, [-1, NUM_OF_CLASSESS + NUM_OF_POINTS])

    return avg_pool

def train(loss_val, learning_rate, global_step, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name='keep_probabilty')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    if FLAGS.mode == 'train':
        filenames, images, widths, heights, num_points, labels = \
            lsun_dataset.read_data(FLAGS.data_dir, 'training', IMAGE_SIZE, FLAGS.batch_size)
        is_train = tf.constant(True)
    elif FLAGS.mode == 'val':
        filenames, images, widths, heights, num_points, labels = \
            lsun_dataset.read_data(FLAGS.data_dir, 'validation', IMAGE_SIZE, FLAGS.batch_size)
        is_train = tf.constant(False)
    else:
        images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input_image')
        labels = tf.placeholder(tf.float32, shape=[None, NUM_OF_CLASSESS + NUM_OF_POINTS], name='label')
        is_train = tf.constant(False)

    logits = inference(images, keep_probability, is_train)
    loss = tf.losses.mean_squared_error(labels, logits)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, learning_rate, global_step, trainable_var)

    print('Setting up summary op...')
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        print('Setting up Saver...')
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored...')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("threads = tf.train.start_queue_runners(coord=coord)")
        step = sess.run(global_step)
        lr = FLAGS.learning_rate

        if FLAGS.mode == 'train':
            for _ in xrange(FLAGS.max_steps - step + 1):
                feed_dict = {keep_probability: 0.5, learning_rate: lr}
                print("sess.run(train_op, feed_dict=feed_dict)")
                sess.run(train_op, feed_dict=feed_dict)

                if step >= int(FLAGS.max_steps * 0.4) and step < int(FLAGS.max_steps * 0.8):
                    lr = FLAGS.learning_rate * 0.5
                elif step >= int(FLAGS.max_steps * 0.8):
                    lr = FLAGS.learning_rate * 0.1

                print("train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)")
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print('Step: %d, Learning rate: %f, Train loss: %f' % (step, lr, train_loss))
                summary_writer.add_summary(summary_str, step)

                if step % 500 == 0:
                    saver.save(sess, FLAGS.logs_dir + 'model.ckpt', step)
                step += 1
        elif  FLAGS.mode == 'val':
            total_type = 0.
            total_points = 0.

            for _ in xrange(10):
                feed_dict = {keep_probability: 1.0}
                pred, gt = sess.run([logits, labels], feed_dict=feed_dict)

                for i in xrange(len(pred)):
                    pred_type = np.argmax(pred[i][:NUM_OF_CLASSESS])
                    gt_type = np.argmax(gt[i][:NUM_OF_CLASSESS])
                    if pred_type == gt_type:
                        total_type += 1
                    total_points += mean_squared_error(pred[i][NUM_OF_CLASSESS:], gt[i][NUM_OF_CLASSESS:])
            print('Type accuracy: %f, Points MSE: %f' % (total_type / (len(pred) * 10),
                total_points / (len(pred) * 10)))
        else:
            num_points = [8, 6, 6, 4, 4, 6, 4, 4, 2, 2, 2]
            files = os.listdir(FLAGS.images_dir)
            random.shuffle(files)

            for file in files:
                org_image = cv2.imread('%s/%s' % (FLAGS.images_dir, file))
                test_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
                test_image = cv2.resize(test_image, (IMAGE_SIZE, IMAGE_SIZE))
                test_image = np.reshape(test_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
                pred = sess.run(logits, feed_dict={images: test_image, keep_probability: 1.0})
                type = np.argmax(pred[0][:NUM_OF_CLASSESS])
                point_pred = pred[0][NUM_OF_CLASSESS:]
                print('Room type: %s, Num of points: %d' % (type, num_points[type]))
                print('point_pred: %s' % point_pred)

                org_image = cv2.resize(org_image, (IMAGE_SIZE, IMAGE_SIZE))
                points = np.reshape(point_pred, (NUM_OF_CELL, NUM_OF_CELL, 3, 3))
                shape = points.shape
                span = 1. / shape[0]

                for y in xrange(shape[0]):
                    for x in xrange(shape[1]):
                        for c in xrange(shape[2]):
                            if points[y][x][c][0] > FLAGS.thresh:
                                point_y = int((y * span + points[y][x][c][1] * span) * IMAGE_SIZE)
                                point_x = int((x * span + points[y][x][c][2] * span) * IMAGE_SIZE)
                                cv2.circle(org_image, (point_x, point_y), 5, (0, 0, 255), -1)
                
                print("Saved to ./out/%s" % (file))
                cv2.imwrite("./out/%s" % (file), org_image)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
