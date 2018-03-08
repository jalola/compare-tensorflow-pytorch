#import test_utils as tu
import torch
import pdb
import tensorflow as tf
from torch import nn
from torch import autograd
import numpy as np
import scipy
import torch.nn.init as init
import gan_pytorch_model as gan
import functools
exec(open("gan_tf_model.py").read())

DEBUG_CONST=5e-2 # All weights are initialized = DEBUG_CONST
import pdb

#np.set_printoptions(precision=20)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight is not None:
            init.constant(m.weight, DEBUG_CONST) 
        if m.bias is not None:
            init.constant(m.bias, 0.0)
    if isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            init.constant(m.weight, 1.0)
        if m.bias is not None:
            init.constant(m.bias, 0.0)


def gen_input(arr):
    return np.around(np.random.rand(*arr), decimals=2)
    #num_ele = int(b * c * h * w)
    #return np.arange(num_ele).reshape(b, c, h, w)
    #return np.full([b, c, h, w], 1.0)

def tf_to_gen_input(*arg):
    arr = [x for x in arg]
    lib.delete_all_params()
    same_input = gen_input(arr)
    tf_input = tf.convert_to_tensor(same_input, np.float32) 
    to_input = autograd.Variable(torch.from_numpy(same_input).type(torch.FloatTensor), requires_grad=True) 
    return (tf_input, to_input)

ERROR_THRESHOLD = 1.5e-4 #Around 1 degree difference, 1 - cosine(degree/180 * pi) 

session = tf.InteractiveSession()

def compare(model, tf_result, to_result):
    print('-----------' + model + '------------')
    tf_np = session.run(tf_result)
    to_np = to_result.data.numpy()
    if model == "ResidualBlock_Up":
        # pdb.set_trace()
        pass
    if model == "GradOfDisc" and to_np.shape == (1,):
        to_np = to_np[0]
        print("GradOfDisc \n tf: {} \n to: {}".format(tf_np, to_np))
    #pdb.set_trace()
    if tf_np.shape == to_np.shape:
        diff_m = np.mean(np.abs(tf_np - to_np))
        m = scipy.spatial.distance.cosine(tf_np.flatten(), to_np.flatten())
        #if model == "Discriminator":
        #    print("tf_np[0]: %.20f" % tf_np[0])
        #    print("to_np[0]: %.20f" % to_np[0])
        #    print("tf_np[1]: %.20f" % tf_np[1])
        #    print("to_np[1]: %.20f" % to_np[1])
        print(str(m < ERROR_THRESHOLD))
        print("tf.abs.mean: " + str(np.abs(tf_np).mean()))
        print("to.abs.mean: " + str(np.abs(to_np).mean()))
        print("diff.mean: " + str(diff_m))
        print('cosine distance of ' + model +': ' + str(m))
    else:
        print('False')
        print('Shape is wrong')
        print('tf: ' + str(tf_np.shape))
        print('to: ' + str(to_np.shape))

def tf_global_init():
    session.run(tf.global_variables_initializer())

b = 512 # Batch
h = 32  # Height
w = 32  # Width
c = 12  # Channel
conv_in = 12
conv_out = 4
fil = 3
dimension = 4 # ?

print("b, c, h, w, in, out: " + str(b) + ", " + str(c) + ", " + str(h) + ", " + str(w) + ", " + str(conv_in) + ", " + str(conv_out))

#TEST
#------------------------------gendata------------------------------
tf_input, to_input = tf_to_gen_input(b, c, h, w)
tf_global_init()
compare('gen_data', tf_input, to_input)

#------------------------------d2s----------------------------------
tf_input, to_input = tf_to_gen_input(b, c, h, w)

tf_output = tf.transpose(tf_input, [0,2,3,1])
tf_output = tf.depth_to_space(tf_output, 2)
tf_output = tf.transpose(tf_output, [0,3,1,2])
tf_global_init()

to_net = gan.DepthToSpace(2)
to_output = to_net(to_input)

compare('depth to space', tf_output, to_output)

#------------------------------conv2d----------------------------------
tf_input, to_input = tf_to_gen_input(b, c, h, w)

tf_output = lib.ops.conv2d.Conv2D("conv2d", conv_in, conv_out, fil, tf_input)
tf_global_init()

to_net = gan.MyConvo2d(conv_in, conv_out, fil)
to_net.apply(weights_init)
to_output = to_net(to_input)

compare('conv2d', tf_output, to_output)

#-----------------------------ConvMeanPool-----------------------------------
tf_input, to_input = tf_to_gen_input(b, c, h, w)

tf_output = ConvMeanPool("ConvMeanPool", conv_in, conv_out, fil, tf_input)
tf_global_init()

to_net = gan.ConvMeanPool(conv_in, conv_out, fil)
to_net.apply(weights_init)
to_output = to_net(to_input)

compare('ConvMeanPool', tf_output, to_output)

#-----------------------------MeanPoolConv-----------------------------------
tf_input, to_input = tf_to_gen_input(b, c, h, w)

tf_output = MeanPoolConv("MeanPoolConv", conv_in, conv_out, fil, tf_input)
tf_global_init()

to_net = gan.MeanPoolConv(conv_in, conv_out, fil)
to_net.apply(weights_init)
to_output = to_net(to_input)

compare('MeanPoolConv', tf_output, to_output)

#-----------------------------UpSampleConv------------------------------------
tf_input, to_input = tf_to_gen_input(b, c, h, w)

tf_output = UpsampleConv("UpsampleConv", conv_in, conv_out, fil, tf_input)
tf_global_init()

to_net = gan.UpSampleConv(conv_in, conv_out, fil)
to_net.apply(weights_init)
to_output = to_net(to_input)

compare('UpsampleConv', tf_output, to_output)

#------------------------------ResidualBlock----------------------------------
#lib.delete_all_params()
tf_input, to_input = tf_to_gen_input(b, c, h, w)

tf_output = ResidualBlock("ResidualBlock", conv_in, conv_out, fil, tf_input, resample='up')
tf_global_init()

to_net = gan.ResidualBlock(conv_in, conv_out, fil, "up")
to_net.apply(weights_init)
to_output = to_net(to_input)

compare('ResidualBlock_Up', tf_output, to_output)

#------------------------------ResidualBlock----------------------------------
#lib.delete_all_params()
tf_input, to_input = tf_to_gen_input(b, c, h, w)

tf_output = ResidualBlock("ResidualBlock", conv_in, conv_out, fil, tf_input, resample='down')
tf_global_init()
to_net = gan.ResidualBlock(conv_in, conv_out, fil, "down")
to_net.apply(weights_init)
to_output = to_net(to_input)

compare('ResidualBlock_Down', tf_output, to_output)

#------------------------------Generator----------------------------------
tf_input, to_input = tf_to_gen_input(64, 128)

tf_output = GoodGenerator(1000, noise=tf_input, dim=dimension)
tf_global_init()
to_net = gan.GoodGenerator(dim=dimension)
to_net.apply(weights_init)
to_output = to_net(to_input)

compare('Generator', tf_output, to_output)

#------------------------------Discriminator----------------------------------
tf_input, to_input = tf_to_gen_input(1024, 3, 64, 64)

compare('D_input', tf_input, to_input)

tf_output = GoodDiscriminator(tf_input, dim=dimension)
tf_global_init()
to_net = gan.GoodDiscriminator(dim=dimension)
to_net.apply(weights_init)
to_output = to_net(to_input)

#pdb.set_trace()
compare('Discriminator', tf_output, to_output)

#------------------------------GradOfDisc----------------------------------
tf_input, to_input = tf_to_gen_input(512, 3 * 64 * 64)

tf_output = TFGradOfDisc(dimension, tf_input) 
tf_global_init()

netD = gan.GoodDiscriminator(dim=dimension)
netD.apply(weights_init)
to_output = gan.TOGradOfDisc(netD, to_input) 

compare('GradOfDisc', tf_output, to_output)

#tf_np = session.run(tf_output)
#to_np = to_output.data.numpy() 

#print("------------GradOfGrad----------")
#print("tf_result: " + str(tf_np))
#print("to_result: " + str(to_np))

diff = session.run(tf_output) - to_output.data.numpy()                                                    
x1 = session.run(tf_output)                                                                               
x2 = to_output.data.numpy() 

#np.argwhere(diff > 0.001)

#-------------------------LayerNorm-Forward-----------------------
import tflib.ops.layernorm
tf_input, to_input = tf_to_gen_input(68, 3, 64, 64)

tf_output = Normalize('Discriminator', [0, 2, 3], tf_input)
tf_global_init()
tf_cost = tf.reduce_mean(tf.square(tf_output))
to_LN = gan.LayerNorm(3) # n_features
to_output = to_LN(to_input)
to_cost = torch.mean(to_output**2)
compare('LayerNorm-Forward', tf_output, to_output)

# Backward
tf_gradients = tf.gradients(tf_cost, [tf_input])[0]
to_gradients = autograd.grad(outputs=to_cost, inputs=to_input, grad_outputs=
                            torch.ones(to_input.size()), create_graph=True, only_inputs=True)[0]

compare('LayerNorm-Backward', tf_gradients, to_gradients)

# Test Support
#x1 = session.run(tf_cost)
#x2 = to_cost.data.numpy()

#grad1 = session.run(tf_gradients)
#grad2 = to_gradients.data.numpy()

#--------------------------BatchNorm--------------------
tf_input, to_input = tf_to_gen_input(68, 3, 64, 64)
n_features = 3

tf_output = tflib.ops.batchnorm.Batchnorm("BN", [0, 2, 3], tf_input, fused=True)
tf_global_init()
tf_cost = tf.reduce_mean(tf.square(tf_output))

to_net = nn.BatchNorm2d(n_features)
to_net.apply(weights_init)
to_output = to_net(to_input)
to_cost = torch.mean(to_output**2)

compare("BatchNorm", tf_output, to_output)


tf_gradients = tf.gradients(tf_cost, [tf_input])[0]
to_gradients = autograd.grad(outputs=to_cost, inputs=to_input, grad_outputs=
                            torch.ones(to_input.size()), create_graph=True, only_inputs=True)[0]

compare('BatchNorm-Backward', tf_gradients, to_gradients)
