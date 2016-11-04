## Kaggle competition Painter by Numbers
<p align="center">
    <img src="/misc/front.jpg?raw=true"/>
</p>

This repository contains a 1st place solution for the Kaggle [competition Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers). Below is a brief description of the dataset and approaches I've used to build and validate a predictive model.

The challenge of the competition was to examine pairs of paintings and determine whether they were painted by the same artist. The training set consists of artwork images and their corresponding class labels (painters). Examples in the test set were split into 13 groups and all possible pairs within each group needed to be examined for the submission. The evaluation metric for the leaderboard was AUC (area under the curve).

### 1. Dataset and preprocessing
The training set is unbalanced and some classes are only present in the training set and some only in the test set. Additionally input images are of various dimensions. There are 79433 instances and 1584 unique painters in the training set and the test set is composed of 23817 instances.  Predictions for approximately 22M pairs needed to be made for the submission.

The plot below shows number of paintings for each of the 1584 painters in the training set.
<p align="center">
    <img src="/misc/num_examples_per_class.png?raw=true"/>
    <b align="center">Number of examples per classes in the training set</b>
</p>

Labeled images were split into training (0.9) and validation (0.1) sets in a stratified manner resulting in 71423 training examples and 8010 validation examples belonging to 1584 classes.

The model I've built assumes a fixed-size inputs, so the first preprocessing step was to resize each image's smallest dimension to 256 pixels (retaining the aspect ratio) and then cropping it at the center of the larger dimension, obtaining 256x256 images. Some information gets lost during this process and an alternative approach where multiple crops are taken from the same image was considered, but not used for the final solution due to much longer training times (bigger, but more correlated training set). Furthermore, mean values were subtracted from each feature in the data and the obtained values were normalized by dividing each dimension by its standard deviation. Preprocessing data statistics were computed from the subset of training instances. During the training phase random transformations (rotations, zooms, shifts, shears and flips) were applied to data in order to reduce overfitting. The latter assures that our model only rarely sees exactly the same example more than once. For exact transformation parameters see [data_provider.py](painters/data_provider.py).

### 2. Building a predictive model
There were two main approaches considered for verifying whether two instances belong to the same class. The unsupervised method involves training a model that can predict one of the 1584 classes and then taking a dot product of the two class distribution vectors (softmax outputs). The supervised method is an end-to-end metric learning approach called siamese network. The main idea is to replicate the model once for each input image and merge their outputs into a single vector, that can then be used to directly predict whether the two images were painted by the same artist. An important aspect of this architecture is that the weights of both models are shared and during backpropagation the total gradient is the sum of the gradients contributed by the two models. Since the model trained for the unsupervised technique can also be used in the siamese architecture, most of the effort went into the multi-class painter recognition task.

Depiction below illustrates the architecture of the final convolutional neural network with non-linearities, dropouts and batch normalization layers omitted. 3x3 convolutional filters with stride 1 are used to produce feature maps, that are two neurons smaller along each of the two dimensions, than their input volumes. Zero padding is then used to retain the original shape and 2x2 max pooling with stride 2 halves the number of neurons along each of the two dimension. Non-linearities are applied to convolution and fully connected outputs using the PReLU function (Leaky ReLU with trainable slope parameter in the negative part). Dense layers at the end of the architecture are the reason why fixed-size inputs need to be fed to the network. The model is regularized using dropout, batch normalization layers and L2 weight penalties. A more detailed architecture and exact values of hyper parameters can be found in [train_cnn.py](painters/train_cnn.py).
```
        LAYER               DATA DIMENSIONS

        Input     #####     (3, 256, 256)
  Convolution      \|/
                  #####     (16, 256, 256)
  Convolution      \|/
                  #####     (16, 256, 256)
   MaxPooling     YYYYY 
                  #####     (16, 128, 128)
  Convolution      \|/  
                  #####     (32, 128, 128)
  Convolution      \|/  
                  #####     (32, 128, 128)
  Convolution      \|/  
                  #####     (32, 128, 128)
   MaxPooling     YYYYY 
                  #####     (32, 64, 64)
  Convolution      \|/  
                  #####     (64, 64, 64)
  Convolution      \|/  
                  #####     (64, 64, 64)
  Convolution      \|/  
                  #####     (64, 64, 64)
   MaxPooling     YYYYY 
                  #####     (64, 32, 32)
  Convolution      \|/
                  #####     (128, 32, 32)
  Convolution      \|/
                  #####     (128, 32, 32)
  Convolution      \|/
                  #####     (128, 32, 32)
   MaxPooling     YYYYY
                  #####     (128, 16, 16)
  Convolution      \|/
                  #####     (256, 16, 16)
  Convolution      \|/
                  #####     (256, 16, 16)
  Convolution      \|/
                  #####     (256, 16, 16)
   MaxPooling     YYYYY
                  #####     (256, 8, 8)
      Flatten     |||||
                  #####     (16384,)
        Dense     XXXXX
                  #####     (2048,)
        Dense     XXXXX
                  #####     (1584,)
      Softmax     #####     (1584,)
```


300 epochs are needed for model to converge to the local minima using the Adam optimizer with 0.000074 learning rate and batch size of 96 examples. During training the cross-entropy loss was minimized.

Neural networks can be used as descriptor generators that produce lower dimensionality representations of input instances. One can think of them as automatic feature extractors. Such embeddings are obtained by simply taking the 2048 dimensional output vectors of the penultimate layer. To check whether there is any internal structure in the features produced by the ConvNet I've used the t-SNE dimensionality reduction technique. t-SNE is a convenient algorithm for visualization of high dimensional data and allows us to compare how similar input instances are. Below are two scatter plots of some of the artwork images of randomly selected artists from the validation set. Having in mind that the network hasn't seen those examples during training and that the t-SNE algorithm doesn't get class labels as inputs, the visual results are quite exciting. For more t-SNE plots see the [misc](misc) directory.
<p align="center">
    <img src="/misc/tsne_3.png?raw=true"/>
    <img src="/misc/tsne_2.png?raw=true"/>
    <b align="center">t-SNE embeddings of the features generated by the ConvNet (click on the image for full resolution)</b>
</p>

### 3. Competition results
The public leaderboard score was calculated on 70% of the submission pairs and the private leaderboard score on the remaining 30%. The final submission was generated using the unsupervised approach for verifying the same class identity. The best single ConvNet scored `0.90717 AUC` on the private leaderboard and an ensemble of 18 best ConvNets trained during the hyper parameter search process scored `0.92890 AUC` on the private leaderboard. Adding more (worse) models to the ensemble started to hurt the overall performance. A single hypothesis was obtained from multiple models as a weighted average of their predictions for the painter recognition task and only then the inner product of the two averaged class distribution vectors was calculated.

The administrator of the competition [Kiri Nichol](https://www.kaggle.com/smallyellowduck) has posted some very useful insights into the performance of the algorithm on the private, test dataset. As stated on the competition [forum](https://www.kaggle.com/c/painter-by-numbers/forums/t/24970/wrapping-up), an ingenious Dutch forger Han van Meegeren was slipped into the test set in order to better understand how good the model is at extracting painters' unique styles. The forger has replicated some of the world's most famous artists' work, including the paintings of Johannes Vermeer. Below is a pairwise comparison table of my best submission's predictions for van Meegeren and Vermeer examples from the test set. Based on the model's predictions it can be seen that Vermeer's paintings are indeed more similar to each other than van Meegeren's paintings are to Vermeer's paintings. It can also be seen that Vermeer's paintings are more similar to each other than van Meegeren's paintings are to each other, due to van Meegeren forging paintings in the style of several different artists.
<p align="center">
    <img src="/misc/vermeer_vs_van_meegeren.png?raw=true"/>
    <b align="center">Pairwise comparison table for van Meegeren and Vermeer paintings in the test set</b>
</p>

Another really valuable insight concerns the extrapolation of the model to artists that were not seen during training. The results are given in the form of AUC of my final submission for two different groups of instances from the test set. The first group consists of pairs of images whose painters were present in the training set: `0.94218 AUC` and the second one is composed of pairs whose artists haven't been seen by the model before: `0.82509 AUC`. The results indicate that the model is not so good at generalizing to unknown classes.

### 4. Conclusion and further work
Based on the results of the competition it can be concluded that convolutional neural networks are able to decompose artwork images' visual space based on their painters unique style. The bad news is that the described algorithm is not good at extrapolating to unfamiliar artists. This is largely due to the fact that same identity verification is calculated directly from the two class distribution vectors.

As my first Kaggle competition this was an excellent learning experience and since I'm planning to continue the work as my upcoming master's degree thesis it was also a great opportunity for me to gain more knowledge about possible pitfalls and challenges in the domain. From this point forward my main focus will be on achieving better generalization by training an end-to-end metric learning technique called siamese network that was only briefly mentioned above.

I would like to thank [Niko Colnerič](https://github.com/nikicc), [Tomaž Hočevar](https://github.com/thocevar), [Blaž Zupan](https://github.com/BlazZupan), [Jure Žbontar](https://github.com/jzbontar) and other members of the [Bioinformatics Laboratory from University of Ljubljana](http://www.biolab.si/en/) for their help and provision of the infrastructure.

### 5. Resources
- [Bioinformatics Laboratory, University of Ljubljana](http://www.biolab.si/en/)
- [Stanford CS231n notes](http://cs231n.github.io)
- Very Deep Convolutional Networks for Large-Scale Image Recognition: Karen Simonyan, Andrew Zisserman
- DeepFace: Closing the Gap to Human-Level Performance in Face Verification: Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf
- Dimensionality Reduction by Learning an Invariant Mapping: Raia Hadsell, Sumit Chopra, Yann LeCun
- Learning a similarity metric discriminatively, with application to face verification: Sumit Chopra, Raia Hadsell, Yann LeCun
- [Keras: Deep Learning library](https://github.com/fchollet/keras)
- [Competition](https://www.kaggle.com/c/painter-by-numbers) and its datasets that were prepared by [Kiri Nichol](https://www.kaggle.com/smallyellowduck)
