import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import MeanPool2d, ExpandDims, Tile, UpSampling2d, Elementwise, \
    GlobalMeanPool2d, InstanceNorm2d, Lambda, Input, Dense, DeConv2d, Reshape,\
    Conv2d, Flatten, Concat, GaussianNoise
from tensorlayer.layers import (SubpixelConv2d, ExpandDims)
from tensorlayer.layers import DeConv2d
from utils import WeightNorm
from config import flags
from tensorlayer.models import Model

w_init = tf.random_normal_initializer(stddev=0.02)
g_init = tf.random_normal_initializer(1., 0.02)
lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)  # tl.act.lrelu(x, 0.2)

def count_weights(model):
    n_weights = 0
    for i, w in enumerate(model.all_weights):
        n = 1
        # for s in p.eval().shape:
        for s in w.get_shape():
            try:
                s = int(s)
            except:
                s = 1
            if s:
                n = n * s
        n_weights = n_weights + n
    print("num of weights (parameters) %d" % n_weights)
    return n_weights

def spectral_norm(w, u, iteration=1):  # https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/spectral_norm.py
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    # u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


class SpectralNormConv2d(Conv2d):
    """
    The :class:`SpectralNormConv2d` class is a Conv2d layer for with Spectral Normalization.
    ` Spectral Normalization for Generative Adversarial Networks (ICLR 2018) <https://openreview.net/forum?id=B1QRgziT-&noteId=BkxnM1TrM>`__

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    dilation_rate : tuple of int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.
    """
    def __init__(
            self,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            data_format='channels_last',
            dilation_rate=(1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None
    ):
        super(SpectralNormConv2d, self).__init__(n_filter=n_filter, filter_size=filter_size,
            strides=strides, act=act, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, W_init=W_init, b_init=b_init, in_channels=in_channels,
            name=name)
        # logging.info(
        #     "    It is a SpectralNormConv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
        #         self.name, n_filter, str(filter_size), str(strides), padding,
        #         self.act.__name__ if self.act is not None else 'No Activation'
        #     )
        # )
        if self.in_channels:
            self.build(None)
            self._built = True

    def build(self, inputs_shape): # # override
        self.u =  self._get_weights("u", shape=[1, self.n_filter], init=tf.random_normal_initializer(), trainable=False) # tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
        # self.s =  self._get_weights("sigma", shape=[1, ], init=tf.random_normal_initializer(), trainable=False)
        super(SpectralNormConv2d, self).build(inputs_shape)

    def forward(self, inputs): # override
        self.W_norm = spectral_norm(self.W, self.u)
        # self.W_norm = spectral_norm(self.W, self.u, self.s)
        # return super(SpectralNormConv2d, self).forward(inputs)
        outputs = tf.nn.conv2d(
            input=inputs,
            filters=self.W_norm,
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,  #'NHWC',
            dilations=self._dilation_rate,  #[1, 1, 1, 1],
            name=self.name,
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs


# architecture: CNN
# discriminator: input X/Y, output likelihood
# def get_D(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), patch_size=(70, 70), name=None):
#     # ref: Image-to-Image Translation with Conditional Adversarial Networks
#     # input: (batch_size_train, 256, 256, 3)
#     # output: (batch_size_train, )
#     df_dim = 64
#
#     nx1 = Input(x_shape)  # RGB
#     n = Lambda(lambda x: tf.image.random_crop(x, [flags.batch_size_train, patch_size[0], patch_size[1], 3]))(nx1)  # patchGAN
#
#     n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, W_init=w_init)(n)
#     n = Conv2d(df_dim * 2, (4, 4), (2, 2), W_init=w_init, b_init=None)(n)
#     n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
#
#     n = Conv2d(df_dim * 4, (4, 4), (2, 2), W_init=w_init, b_init=None)(n)
#     n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
#
#     n = Conv2d(df_dim * 8, (4, 4), (2, 2), W_init=w_init, b_init=None)(n)
#     n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
#     # print("X"*100, n.shape)
#     # exit()
#     if patch_size == (70, 70):
#         n = Conv2d(1, (4, 4), (4, 4), padding='VALID', W_init=w_init)(n)
#     elif patch_size == (140, 140):
#         # n = Conv2d(X, (X, X), (X, X), W_init=w_init)(n)
#         # n = Conv2d(1, (X, X), (X, X), padding='VALID', W_init=w_init)(n)
#         raise Exception("Unimplement: TODO")
#     else:
#         raise Exception("Unknown patch_size")
#
#     n = Flatten()(n)
#     assert n.shape[-1] == 1  # check
#     M = Model(inputs=nx1, outputs=n, name=name)
#     return M


def get_D(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name=None):
    # ref: Image-to-Image Translation with Conditional Adversarial Networks
    # input: (batch_size_train, 256, 256, 3)
    # output: (batch_size_train, )
    ch = 64
    n_layer = 8
    tch = ch
    ni = Input(x_shape)
    n = SpectralNormConv2d(ch, (3, 3), (2, 2), act=lrelu, W_init=w_init)(ni)
    for i in range(1, n_layer-1):
        n = SpectralNormConv2d(tch * 2, (3, 3), (2, 2), act=lrelu, W_init=w_init)(n)
        tch *= 2
    n = SpectralNormConv2d(tch * 2, (3, 3), (2, 2), act=lrelu, W_init=w_init)(n)
    tch *= 2
    n = SpectralNormConv2d(1, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init)(n)
    n = Reshape([-1, 1])(n)
    M = Model(inputs=ni, outputs=n, name=name)
    return M


# architecture: CNN
# content encoder: input X, output c
# def get_Ec(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name=None):
#     # ref: Multimodal Unsupervised Image-to-Image Translation
#     # input: (batch_size_train, 256, 256, 3)
#     # output: 3d tensor (batch_size_train, 64, 64, 256)
#     w_init = tf.random_normal_initializer(stddev=0.02)
#
#     ni = Input(x_shape)
#     n = Conv2d(64, (7, 7), (1, 1), act=tf.nn.relu, W_init=w_init)(ni)
#     n = Conv2d(128, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
#     n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
#
#     nn = Conv2d(256, (3, 3), (1, 1), act=tf.nn.relu, W_init=w_init)(n)
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init)(nn)
#     n = Elementwise(tf.add)([n, nn])
#
#     nn = Conv2d(256, (3, 3), (1, 1), act=tf.nn.relu, W_init=w_init)(n)
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init)(nn)
#     n = Elementwise(tf.add)([n, nn])
#
#     nn = Conv2d(256, (3, 3), (1, 1), act=tf.nn.relu, W_init=w_init)(n)
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init)(nn)
#     n = Elementwise(tf.add)([n, nn])
#
#     nn = Conv2d(256, (3, 3), (1, 1), act=tf.nn.relu, W_init=w_init)(n)
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init)(nn)
#     n = Elementwise(tf.add)([n, nn])
#
#     M = Model(inputs=ni, outputs=n, name=name)
#     return M


def get_Ec(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name=None):
    # ref: Multimodal Unsupervised Image-to-Image Translation
    lrelu = lambda x: tl.act.lrelu(x, 0.01)
    w_init = tf.random_normal_initializer(stddev=0.02)
    channel = 64
    ni = Input(x_shape)
    n = Conv2d(channel, (7,7), (1,1), act=lrelu, W_init=w_init)(ni)
    for i in range(2):
        n = Conv2d(channel * 2, (3, 3), (2, 2), W_init=w_init)(n)
        n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)
        channel = channel * 2

    for i in range(1, 5):
        # res block
        nn = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
        nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
        nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
        n = Elementwise(tf.add)([n, nn])

    n = GaussianNoise(is_always=False)(n)

    M = Model(inputs=ni, outputs=n, name=name)
    return M


# architecture: VAE encoder
# appearance encoder: input X, output
def get_Ea(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name=None): # input: (1, 256, 256, 3)
    # ref: DRIT source code (Pytorch Implementation)
    w_init = tf.random_normal_initializer(stddev=0.02)
    ndf = 64
    ni = Input(x_shape)
    nn = Conv2d(ndf, (7, 7), (1, 1), padding='VALID', W_init=w_init, act=None)(ni)

    ## Basic Blocks * 3
    for i in range(1, 4):
        ## Basic Block
        # conv part
        n = Lambda(lambda x: tf.nn.leaky_relu(x, 0.2))(nn)  # leaky relu (0.2)
        n = Conv2d(ndf * i, (3, 3), (1, 1), padding='VALID', W_init=w_init, act=lrelu)(n)  # conv3x3
        n = Conv2d(ndf * (i + 1), (3, 3), (1, 1), padding='VALID', W_init=w_init, act=None)(n)  # conv3x3 in convMeanpool
        n = MeanPool2d((2, 2), (2, 2))(n)  # meanPool2d in convMeanpool
        # shortcut part
        ns = MeanPool2d((2, 2), (2, 2))(nn)
        ns = Conv2d(ndf * (i + 1), (3, 3), (1, 1), padding='VALID', W_init=w_init)(ns)
        nn = Elementwise(tf.add)([n, ns])

    n = GlobalMeanPool2d()(nn)
    n_mu = Dense(n_units=flags.za_dim, W_init=w_init, name="mean_linear")(n)
    n_log_var = Dense(n_units=flags.za_dim, W_init=w_init, name="var_linear")(n)

    # Sampling using reparametrization trick
    def sample(input):
        n_mu = input[0]
        n_log_var = input[1]
        epsilon = tf.random.truncated_normal(n_mu.shape)
        stddev = tf.exp(n_log_var)
        out = n_mu + stddev * epsilon
        return out

    no = Lambda(sample)([n_mu, n_log_var])
    M = Model(inputs=ni, outputs=[no, n_mu, n_log_var], name=name)
    return M

#
# # architecture: Conv2d + Res blocks + Upsampling
# # generator: input c,z, output X/Y
# def get_G(z_shape=(None, flags.za_dim), c_shape=(None, flags.c_shape[0], \
#         flags.c_shape[1], flags.c_shape[2]), name=None):
#     nc = Input(c_shape)
#     nz = Input(z_shape)
#     n = ExpandDims(1)(nz)
#     n = ExpandDims(1)(n)
#     n = Tile([1, c_shape[1], c_shape[2], 1])(n)
#     n = Concat(-1)([nc, n])
#
#     # 264->256 多一层??
#     n = Conv2d(256, (1, 1), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
#
#     # res blocks
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#     nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
#     n = Elementwise(tf.add)([n, nn])
#
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#     nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
#     n = Elementwise(tf.add)([n, nn])
#
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#     nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
#     n = Elementwise(tf.add)([n, nn])
#
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
#     nn = Conv2d(256, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#     nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
#     n = Elementwise(tf.add)([n, nn])
#
#     # deconvs
#     n = UpSampling2d((2, 2))(n)
#     n = Conv2d(128, (5, 5), (1, 1), W_init=w_init, b_init=None, act=None)(n)
#     n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
#
#     n = UpSampling2d((2, 2))(n)
#     n = Conv2d(128, (5, 5), (1, 1), W_init=w_init, b_init=None, act=None)(n)
#     n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
#
#     n = Conv2d(3, (7, 7), (1, 1), W_init=w_init, b_init=None, act=tf.nn.tanh)(n)
#
#     M = Model(inputs=[nz, nc], outputs=n, name=name)
#     return M
#

# architecture: dcgan (z->img) -- (z->content tensor)
# input zc:(batch_size_train, 100)
# output content tensor: (batch_size_train, 64, 64, 256)


# generator: input a, c, txt, output X', encode 64*64
# generator: input a, c, txt, output X', encode 64*64
def get_G(a_shape=(None, flags.za_dim), c_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]), \
          name=None):
    ndf = 256
    na = Input(a_shape)
    nc = Input(c_shape)
    #z = Concat(-1)([na, nt])
    z = na
    nz = ExpandDims(1)(z)
    nz = ExpandDims(1)(nz)
    nz = Tile([1, c_shape[1], c_shape[2], 1])(nz)

    # res block
    nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nc)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
    nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
    n = Elementwise(tf.add)([nc, nn])

    nd_tmp = flags.za_dim
    ndf = ndf + nd_tmp
    n = Concat(-1)([n, nz])

    # res block *3
    for i in range(1, 4):
        nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
        nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
        nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
        n = Elementwise(tf.add)([n, nn])

    for i in range(2):
        ndf = ndf + nd_tmp
        n = Concat(-1)([n, nz])
        nz = Tile([1, 2, 2, 1])(nz)

        n = DeConv2d(ndf // 2, (3, 3), (2, 2), act=tf.nn.relu, W_init=w_init, b_init=None)(n)
        n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)

        ndf = ndf // 2

    n = Concat(-1)([n, nz])
    n = DeConv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, W_init=w_init)(n)

    M = Model(inputs=[na, nc], outputs=n, name=name)
    return M


def get_G_zc(shape_z=(None, flags.zc_dim), gf_dim=64):
    # reference: DCGAN generator
    output_size = 64
    s16 = output_size // 16

    ni = Input(shape_z)
    nn = Dense(n_units=(gf_dim * 8 * s16 * s16), W_init=w_init, b_init=None)(ni)
    nn = Reshape(shape=[-1, s16, s16, gf_dim * 8])(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(gf_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(gf_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(gf_dim, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(256, (5, 5), (2, 2), act=tf.nn.tanh, W_init=w_init)(nn)

    return tl.models.Model(inputs=ni, outputs=nn, name='Generator_zc')


# architecture: CNN
# input: (batch_size_train, 256, 256, 3) output: (batch_size_train, 100)
def get_E_x2zc(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name="Encoder_x2zc"):
    # ref: Multimodal Unsupervised Image-to-Image Translation
    # input: (batch_size_train, 256, 256, 3)
    # output: vector (batch_size_train, za_dim)
    w_init = tf.random_normal_initializer(stddev=0.02)
    ni = Input(x_shape)
    n = Conv2d(64, (7, 7), (1, 1), act=tf.nn.relu, W_init=w_init)(ni)
    n = Conv2d(128, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = GlobalMeanPool2d()(n)
    n = Flatten()(n)
    n = Dense(flags.zc_dim)(n)
    M = Model(inputs=ni, outputs=n, name=name)
    return M


# architecture: CNN
# input: (batch_size_train, 256, 256, 3) output: (batch_size_train, 100)
def get_E_x2za(x_shape=(None, flags.img_size_h, flags.img_size_w, flags.c_dim), name="Encoder_x2za"):
    # ref: Multimodal Unsupervised Image-to-Image Translation
    # input: (batch_size_train, 256, 256, 3)
    # output: vector (batch_size_train, za_dim)
    w_init = tf.random_normal_initializer(stddev=0.02)
    ni = Input(x_shape)
    n = Conv2d(64, (7, 7), (1, 1), act=tf.nn.relu, W_init=w_init)(ni)
    n = Conv2d(128, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = Conv2d(256, (4, 4), (2, 2), act=tf.nn.relu, W_init=w_init)(n)
    n = GlobalMeanPool2d()(n)
    n = Flatten()(n)
    n = Dense(flags.za_dim)(n)
    M = Model(inputs=ni, outputs=n, name=name)
    return M


# architecture: CNN
# input content tensor: (batch_size_train, 64, 64, 256)
# output: likelihood
def get_D_content(c_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2])):
    # reference: DRIT resource code -- Pytorch implementation
    ni = Input(c_shape)
    n = Conv2d(256, (7, 7), (2, 2), act=None, W_init=w_init)(ni)
    n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
    n = Conv2d(256, (7, 7), (2, 2), act=None, W_init=w_init)(n)
    n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
    n = Conv2d(256, (7, 7), (2, 2), act=None, W_init=w_init)(n)
    n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
    n = Conv2d(256, (4, 4), (1, 1), act=None, padding='VALID', W_init=w_init)(n)
    n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
    n = Conv2d(1, (5, 5), (5, 5), padding='VALID', W_init=w_init)(n)
    n = Reshape(shape=[-1, 1])(n)
    return tl.models.Model(inputs=ni, outputs=n, name=None)


if __name__ == '__main__':


    x = tf.zeros([1, 256, 256, 3])
    E_xa = get_Ea(x.shape)
    E_xa.train()
    print("Ea:"+str(count_weights(E_xa)))

    xa, xa_mu, xa_var = E_xa(x)  # (1, 8)
    # # print(a_vec.shape)
    # # print(xa_mu.shape)
    # # print(xa_var.shape)
    # # print(xa.shape)
    #
    D = get_D(x.shape)
    D.train()
    print("D:"+str(count_weights(D)))
    # label = D(x)
    # # print(label.shape)
    E_xc = get_Ec(x.shape)
    E_xc.train()
    xc = E_xc(x)
    print("Ec:"+str(count_weights(E_xc)))
    # # print(xc.shape)
    G = get_G(xa.shape, xc.shape)
    G.train()
    print("G:"+str(count_weights(G)))

    img_x = G([xa, xc])  # (1, 256, 256, 3)
    # print(img_x.shape)
    #
    zc = tf.zeros([1, 100])
    G_zc = get_G_zc(zc.shape)
    G_zc.train()
    print("G_zc:"+str(count_weights(G_zc)))
    # z_content = G_zc(zc)  # (1, 64, 64, 256)
    # # print(z_content.shape)
    #
    za = tf.zeros([1, 8])
    G_y = get_G(za.shape, z_content.shape)
    G_y.train()
    print("G_y"+str(count_weights(G_y)))
    # img_y = G_y([za, z_content])  # (1, 256, 256, 3)
    # # print(img_y.shape)
    #
    img_y = tf.zeros([1, 216, 216, 3])
    E_y_zc = get_E_x2zc(img_y.shape)
    E_y_za = get_E_x2za(img_y.shape)
    E_y_zc.train()
    E_y_za.train()
    print("E_y_zc"+str(count_weights(E_y_zc)))
    print("E_y_za"+str(count_weights(E_y_za)))
    # zc_ = E_y_zc(img_y)  # (1, 100)
    # za_ = E_y_za(img_y)  # (1, 8)
    # # print(zc.shape)
    # # print(za.shape)
    #

    # label_x = D_x(img_x)  # (1, 1)
    # label_y = D_y(img_y)  # (1, 1)
    # # print(label_x.shape)
    # # print(label_y.shape)
    #
    D_content = get_D_content(xc.shape)
    D_content.train()
    print("D_c"+str(count_weights(D_content)))
    # label_c_x = D_content(xc)  # (1, 1)
    # label_c_zc = D_content(z_content)  # (1, 1)
    # # print(label_c_x.shape)
    # # print(label_c_zc.shape)

    # y = tf.zeros([1, 256, 256, 3])
    # E_ya = get_E_ya(y.shape)
    # E_ya.train()  # (1, 64)
    # ya = E_ya(y)
    # print(ya.shape)
    # y = tf.zeros([1, 256, 256, 3])
    # E_yc = get_E_yc(y.shape)
    # E_yc.train()  # (1, 64, 64, 256)
    # yc = E_yc(y)
    # print(yc.shape)
