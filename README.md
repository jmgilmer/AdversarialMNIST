# Generating Adversarial Images from the MNIST Dataset

Recently there have been a number of papers which have shown that many common neural network architectures trained for image recognition are vulnerable to "adversarial images", see [Intriguing properties of neural networks [1]](http://arxiv.org/abs/1312.6199), and [Explaining and Harnessing Adversarial Examples [2]](http://arxiv.org/pdf/1412.6572v3.pdf). This project is aimed to explore some of the ideas in these papers, and to test a hypothesis of my own. We start first with the [deep CNN model given in the tensorflow tutorial](https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html) and try to generate adversarial images for it. We are able to fool the model on 40% of the samples in the test set by perturbing each image pixel by <= .1 (image pixels are in the range of 0 to 1). We then try to see if L1-regularization can help the model become more robust to image perturbation. Initial results however suggest that the L1-regularized model is **more sensitive** to adversarial images than the baseline.

## What is an Adversarial Image?
[2] found a simple way to fool a neural network trained for image recognition. If you take the gradient of the loss function with respect to the input image and the perturb the input by small amounts in the direction of the gradient, the resulting image is often misclassified by the classifier. In the example below we start with an image which is correctly classified as an 8 with 99.9% probability, however by changing each pixel by .1 in the direction of the gradient we arrive at an image which is classified as a 5 with 99.9% probability:

![alt-text](http://i.imgur.com/RVZdZOh.jpg)

I found on MNIST one has to perturb pixels by at least .05 to fool the network. Other datasets with larger inputs might work with smaller perturbations.

## Motivation for L1-regularization
[2] gave a simple proposition for why neural networks are vulnerable to adversarial images. Imagine a single neuron in a fully connected layer with weight vector w and input x. The neuron computes w dot x, and then applies a non-linearity to the result. If we perturb x by epsilon*sign(w), then the activation in the neuron will increase by epsilon * ||w||, where ||w|| is the l1-norm of w. If w has large l1-norm than this activation can change by a lot. This implies it is easy to change some activations at least in the first layer of the network, perhaps it is also possible to change the activations of the output of the network. This same logic applies to CNN models, as the convolutions are also linear maps. I hoped by adding in an L1 term to the loss function, I could bias the final model towards one where internal nodes and convolutions had smaller L1 norms and thus the result would be more robust to adversarial images. It turns out it was easy to force smaller weights within the network while maintaining good accuracy on the test set, but to my surprise the resulting network was actually more sensitive to adversarial samples.

## Results
The baseline model minimizes the cross entropy L(W), the l1-regularized model minimizes L(W) + 1/20000 * || W ||, where ||W|| is the l1 norm of all of the weights in the model. The trained baseline model had total L1 norm of 230000, the trained regularized model had L1 norm  of 28000. Both models got ~99.2% accuracy on the test set. For most images the regularized model has much smaller gradients (with respect to image pixels) than the baseline model. The plot below (note log scale of y-axis) shows the percentiles of the log of the l1 norms of the gradients.

![alt-text](http://i.imgur.com/pJLY3N5.png)

It appears the regularization has pushed the size of the gradients much closes to 0. However, for some images it has much bigger gradients, in particular for 1000 images the gradient computation returns nan! I suspect this is due to exploding gradient, but need to look closer at the activations to be sure. Even though for most images the gradients are smaller, the regularized model is much more sensitive to perturbations, see the plot below. Epsilon indicates the amount each image pixel is perturbed by in the direction of the gradient.

![alt-text](http://i.imgur.com/6XZBbXi.png)

 Perhaps the error surface of the l1 model is more extreme than baseline, where gradients tend to be either very flat or very extreme. 



> Written with [StackEdit](https://stackedit.io/).