# Compare outputs of Wasserstein GANs between TensorFlow vs Pytorch

This is our testing module for the implementation of [improved WGAN in Pytorch](https://github.com/jalola/improved-wgan-pytorch)


# Prerequisites
* Python >= 3.6
* Pytorch [Latest version from master branch](https://github.com/pytorch/pytorch)
* Numpy
* SciPy
* TensorFlow

# How to run

Go to `test` directory and run ```python test_compare_tf_to.py```

# How we do it

We inject the same weights init and inputs into layers of TensorFlow and Pytorch that we want to compare. For example, we set 5e-2 for the weights of Conv2d layer in both TensorFlow and Pytorch. Then we passed the same random input to those 2 layers and finally we compared 2 outputs from TensorFlow tensor and Pytorch tensor.

We use cosine to calculate the distance between 2 outputs. Reference: [scipy.spatial.distance.cosine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)

# What were compared between TensorFlow and Pytorch

We've compared the implementation of several layers in WGAN model. They are:
* Depth to space
* Conv2d
* ConvMeanPool
* MeanPoolConv
* UpsampleConv
* ResidualBlock (up)
* ResidualBlock (down)
* GoodGenerator 
* Discriminator 
* LayerNorm
* BatchNorm
* Gradient of Discriminator
* Gradient of LayerNorm
* Gradient of BatchNorm

# Result

There are some weird results (cosine < 0 or the distance is bigger than defined threshold - 1 degree) and we look forward to your comments. Here are the outputs of the comparison.

b, c, h, w, in, out: 512, 12, 32, 32, 12, 4  

-----------gen_data------------  
True  
tf.abs.mean: 0.500134  
to.abs.mean: 0.500134  
diff.mean: 0.0  
cosine distance of gen_data: 0.0  

-----------depth to space------------  
True  
tf.abs.mean: 0.500047  
to.abs.mean: 0.500047  
diff.mean: 0.0
cosine distance of depth to space: 0.0  

-----------conv2d------------  
True  
tf.abs.mean: 2.5888  
to.abs.mean: 2.5888  
diff.mean: 3.56939e-07  
cosine distance of conv2d: 5.96046447754e-08  

-----------ConvMeanPool------------  
True  
tf.abs.mean: 2.58869  
to.abs.mean: 2.58869  
diff.mean: 2.93676e-07  
cosine distance of ConvMeanPool: 0.0  

-----------MeanPoolConv------------  
True  
tf.abs.mean: 2.48026  
to.abs.mean: 2.48026  
diff.mean: 3.42314e-07  
cosine distance of MeanPoolConv: 0.0  

-----------UpsampleConv------------  
True  
tf.abs.mean: 2.64478  
to.abs.mean: 2.64478  
diff.mean: 5.50668e-07  
cosine distance of UpsampleConv: 0.0  

-----------ResidualBlock_Up------------  
True  
tf.abs.mean: 1.01438  
to.abs.mean: 1.01438  
diff.mean: 5.99736e-07  
cosine distance of ResidualBlock_Up: 0.0  

-----------ResidualBlock_Down------------  
False  
tf.abs.mean: 2.38841  
to.abs.mean: 2.38782  
diff.mean: 0.192403  
cosine distance of ResidualBlock_Down: 0.00430130958557  

-----------Generator------------  
True  
tf.abs.mean: 0.183751  
to.abs.mean: 0.183751  
diff.mean: 9.97704e-07  
cosine distance of Generator: 0.0  

-----------D_input------------  
True  
tf.abs.mean: 0.500013  
to.abs.mean: 0.500013  
diff.mean: 0.0  
cosine distance of D_input: 0.0  

-----------Discriminator------------  
True  
tf.abs.mean: 295.795  
to.abs.mean: 295.745  
diff.mean: 0.0496472  
cosine distance of Discriminator: 0.0  

-----------GradOfDisc------------  
GradOfDisc  
 tf: 315944.9375  
 to: 315801.09375  
True  
tf.abs.mean: 315945.0  
to.abs.mean: 315801.0  
diff.mean: 143.844  
cosine distance of GradOfDisc: 0.0  

-----------LayerNorm-Forward------------  
True  
tf.abs.mean: 0.865959  
to.abs.mean: 0.865946  
diff.mean: 1.3031e-05  
cosine distance of LayerNorm-Forward: -2.38418579102e-07  

-----------LayerNorm-Backward------------  
False  
tf.abs.mean: 8.67237e-10  
to.abs.mean: 2.49221e-10  
diff.mean: 6.18019e-10  
cosine distance of LayerNorm-Backward: 0.000218987464905  

-----------BatchNorm------------  
True  
tf.abs.mean: 0.865698  
to.abs.mean: 0.865698  
diff.mean: 1.13394e-07  
cosine distance of BatchNorm: 0.0  

-----------BatchNorm-Backward------------  
True  
tf.abs.mean: 8.66102e-10  
to.abs.mean: 8.62539e-10  
diff.mean: 3.56342e-12  
cosine distance of BatchNorm-Backward: 4.17232513428e-07  

# Acknowledge

* [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
* [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
* [LayerNorm](https://github.com/pytorch/pytorch/issues/1959)