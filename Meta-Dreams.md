# Meta-Dreams

* Solves: 15
* Points: 1750
* Category: AI

In 2015, computer scientists made Deep Dreams. I dreamed of the arctic pandas. They could reconstruct the style of anything.

But now, I've been dreaming in color- starting from the farthest, blackest corners of my imagination. Can you tell me which one?

#### Hint
Flag is to be submitted in hex:

FFFFFF or rtcp{FFFFFF}

## Challenge

We were given a file : `Dream.pth`

As you may know, pth files are used to save data for pytorch' models. This type of file can either contain the whole model with its weights or just a python OrderedDict containing the weights : the state_dictionary of a pytorch model.

## Solution

In this case, there is only the OrderedDict in `Dream.pth` aka the state_dictionary:
```python
>>> import torch
>>> dict = torch.load("Dream.pth")
>>> print(type(dict))
<class 'collections.OrderedDict'>
```

So, our goal is to create the model corresponding to these weights, load the weights into the model and feed it a black image. The flag is the color of this output image in hex.

The structure of the model can be retrieved from the state_dictionary:
```python
>>> for i in dict:
...     print(i)
...
conv1.conv2d.weight
conv1.conv2d.bias
in1.weight
in1.bias
conv2.conv2d.weight
conv2.conv2d.bias
in2.weight
in2.bias
conv3.conv2d.weight
conv3.conv2d.bias
in3.weight
in3.bias
res1.conv1.conv2d.weight
res1.conv1.conv2d.bias
res1.in1.weight
res1.in1.bias
res1.conv2.conv2d.weight
res1.conv2.conv2d.bias
res1.in2.weight
res1.in2.bias
[...]
res5.in2.bias
deconv1.conv2d.weight
deconv1.conv2d.bias
in4.weight
in4.bias
deconv2.conv2d.weight
deconv2.conv2d.bias
in5.weight
in5.bias
deconv3.conv2d.weight
deconv3.conv2d.bias
```

So we're dealing with a custom neural network that performs convolution and deconvolution on the input.
This model doesn't seem to be a "classical" model that we can simply import from pytorch library.

By looking into the challenge creator's github repo, we find that one of his projects : 
https://github.com/JEF1056/Reconstruction-Style contains the structure of this specific convolutional neural network.

We can copy his model code to create the model and import the weights.

With numpy and PIL, we create a 300x300 black image:
```python
>>> import numpy as np
>>> from PIL import Image
>>> im_array = np.zeros((300,300,3))
>>> Image.fromarray(im_array.astype('uint8')).save("input.png")
```

We then feed it into the model and extract the color:
```python
def test(content_size):
"""Stylize a content image"""

	transformer = TransformerNet()
    transformer.load_state_dict(torch.load('Dream.pth'))

    content_transform = transforms.Compose([
		transforms.Resize(content_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))])
    content_image = load_image("input.png")
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    output = transformer(content_image).cpu().detach()
    save_image("output.png", output[0] * 255)

test(300)

im = Image.open("output.png")
arr = np.asarray(im)
flag = "rtcp{"
for i in arr[0,0,:]:
	flag += hex(i)[2:]
flag += "}"
print(flag)
```

Flag:
```
rtcp{78857c}
```
