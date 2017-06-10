import os
import argparse
import numpy as np
import tensorflow as tf
import functools
import tensorflow.contrib.slim as slim
import scipy.misc
from align import align_face


class ImageUnreadableException(Exception):
    message = 'The given file could not be read as an image file.'


class NoFaceDetectedException(Exception):
    message = 'No face could be detected in your image.'


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """
    transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    """
    if dtype is None:
        dtype = images.dtype
    return (
        (images + 1.) / 2. * (max_value - min_value) + min_value
    ).astype(dtype)


def imread(path, mode='RGB'):
    """
    read an image into [-1.0, 1.0] of float64
    """
    return scipy.misc.imread(path, mode=mode) / 127.5 - 1


def imwrite(image, path, **kwargs):
    """ save an [-1.0, 1.0] image """
    image = np.array(image)
    image = ((image + 1.) / 2. * 255).astype(np.uint8)
    return scipy.misc.imsave(path, image, **kwargs)


def im2uint(images):
    """ transform images from [-1.0, 1.0] to uint8 """
    return to_range(images, 0, 255, np.uint8)


def imresize(image, size, interp='bilinear'):
    """
    Resize an [-1.0, 1.0] image.

    size : int, float or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp : str, optional
        Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
        'bicubic' or 'cubic').
    """

    # scipy.misc.imresize should deal with uint8 image, or it would cause some
    # problem (scale the image to [0, 255])
    return (
        scipy.misc.imresize(im2uint(image), size, interp=interp) / 127.5 - 1
    ).astype(image.dtype)


conv = functools.partial(slim.conv2d, activation_fn=None)
deconv = functools.partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu


def generator(img, scope, gf_dim=64, reuse=False, train=True):
    bn = functools.partial(
        slim.batch_norm,
        scale=True,
        is_training=train,
        decay=0.9,
        epsilon=1e-5,
        updates_collections=None
    )

    def residule_block(x, dim, scope='res'):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = relu(bn(
            conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv1'),
            scope=scope + '_bn1'
        ))
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = bn(
            conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv2'),
            scope=scope + '_bn2'
        )
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        c0 = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = relu(bn(
            conv(c0, gf_dim, 7, 1, padding='VALID', scope='c1_conv'),
            scope='c1_bn'
        ))
        c2 = relu(bn(
            conv(c1, gf_dim * 2, 3, 2, scope='c2_conv'),
            scope='c2_bn'
        ))
        c3 = relu(bn(
            conv(c2, gf_dim * 4, 3, 2, scope='c3_conv'), scope='c3_bn'
        ))

        r1 = residule_block(c3, gf_dim * 4, scope='r1')
        r2 = residule_block(r1, gf_dim * 4, scope='r2')
        r3 = residule_block(r2, gf_dim * 4, scope='r3')
        r4 = residule_block(r3, gf_dim * 4, scope='r4')
        r5 = residule_block(r4, gf_dim * 4, scope='r5')
        r6 = residule_block(r5, gf_dim * 4, scope='r6')
        r7 = residule_block(r6, gf_dim * 4, scope='r7')
        r8 = residule_block(r7, gf_dim * 4, scope='r8')
        r9 = residule_block(r8, gf_dim * 4, scope='r9')

        d1 = relu(bn(
            deconv(r9, gf_dim * 2, 3, 2, scope='d1_dconv'),
            scope='d1_bn'
        ))
        d2 = relu(bn(
            deconv(d1, gf_dim, 3, 2, scope='d2_dconv'),
            scope='d2_bn'
        ))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv(d2, 3, 7, 1, padding='VALID', scope='pred_conv')
        pred = tf.nn.tanh(pred)

        return pred


class Model:
    def __init__(self, checkpoint_dir, crop_size=256):
        self.session = tf.Session()
        self.checkpoint_dir = checkpoint_dir
        self.crop_size = crop_size

        self.a_real = tf.placeholder(
            tf.float32, shape=[None, crop_size, crop_size, 3]
        )
        self.b_real = tf.placeholder(
            tf.float32, shape=[None, crop_size, crop_size, 3]
        )

        self.a2b = generator(self.a_real, 'a2b')
        self.b2a = generator(self.b_real, 'b2a')
        self.b2a2b = generator(self.b2a, 'a2b', reuse=True)
        self.a2b2a = generator(self.a2b, 'b2a', reuse=True)

        self.load_checkpoint(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir):
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if not ckpt or not ckpt.model_checkpoint_path:
            raise FileNotFoundError

        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        saver.restore(self.session, ckpt_path)

    def run(self, direction, image):
        assert direction in ['a2b', 'b2a']
        size = self.crop_size

        if direction == 'a2b':
            in_tensor = self.a_real
            model = self.a2b
        else:
            in_tensor = self.b_real
            model = self.b2a

        import time

        t0 = time.clock()
        aligned_image = align_face(image, size)
        print('Aligning took', time.clock() - t0, 'seconds')

        if aligned_image is None:
            raise NoFaceDetectedException()

        t0 = time.clock()
        real_ipt = imresize(aligned_image, [size, size])
        real_ipt.shape = 1, size, size, 3
        print('Resizing took', time.clock() - t0, 'seconds')

        t0 = time.clock()
        result = self.session.run(model, feed_dict={in_tensor: real_ipt})[0]
        print('Model took', time.clock() - t0, 'seconds')

        return result

    def run_on_filepath(self, direction, input_path, output_path):
        try:
            image = imread(input_path)
        except OSError:
            raise ImageUnreadableException()

        output = self.run(direction, image)
        imwrite(output, output_path)

    def run_on_filedescriptor(
        self, direction, inputfile, outputfile, format='JPEG'
    ):
        try:
            image = imread(inputfile)
        except OSError:
            raise ImageUnreadableException()

        output = self.run(direction, image)
        imwrite(output, outputfile, format=format)

    def close(self):
        self.session.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply a trained CycleGAN to a picture'
    )
    parser.add_argument(
        'checkpoint_dir', type=str, help='Checkpoint dir to load model from'
    )
    parser.add_argument(
        'direction', type=str, choices=['a2b', 'b2a'],
        help='The direction of the model to apply'
    )
    parser.add_argument(
        'input_path', type=str, help='Input image path'
    )
    parser.add_argument(
        'output_path', type=str, help='Output image path'
    )
    parser.add_argument(
        '--crop_size', dest='crop_size', type=int, default=256,
        help='then crop to this size'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = None
    try:
        model = Model(args.checkpoint_dir, crop_size=args.crop_size)

        model.run_on_filepath(
            args.direction, args.input_path, args.output_path
        )

    finally:
        if model:
            model.close()


if __name__ == '__main__':
    main()
