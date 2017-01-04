# Maxout Networks (Ian Goodfellow, Yoshua Bengio - 2013)
Maxout Networks TensorFlow implementation presented in https://arxiv.org/abs/1302.4389

<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" width="150"><br><br>
</div>
## How to clone the repo?
```
git clone git@github.com:philipperemy/tensorflow-maxout.git maxout
cd maxout
# make sure Tensorflow is installed.
python maxout.py # to see how it's usable.
```

## How to integrate it in your code

It's two lines of code. Sorry I can't make it shorter.

```
from maxout import max_out
t = max_out(tf.matmul(x, W1) + b1, num_units=50)
```

# Some Results on MNIST dataset

Those results are not meant to reproduce the results of the paper. It's more about showing on how to use the maxout non linearity in the Tensorflow graphs.

## Loss
<div align="center">
  <img src="fig/mnist.png" width="400"><br><br>
</div>
