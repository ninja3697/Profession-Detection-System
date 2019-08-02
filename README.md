# Profession-Detection-System

**Introduction**

One of the important fields of Artificial Intelligenceis Computer Vision. Computer Vision is the science of computers and software systems that can recognize and understand images and scenes. Computer Vision is also composed of various aspects such as image recognition, object detection, image generation, image super-resolution and more.
The breakthrough and rapid adoption of deep learning in 2012 brought into existence modern and highly accurate object detection algorithms and methods such as R-CNN, RetinaNet and fast yet highly accurate ones like SSD and YOLO. Using these methods and algorithms, based on deep learning which is also based on machine learning require lots of mathematical and deep learning frameworks understanding.
CNNs can be thought of automatic feature extractors from the image. While if we use a algorithm with pixel vector we lose a lot of spatial interaction between pixels, a CNN effectively uses adjacent pixel information to effectively downsample the image first by convolution and then uses a prediction layer at the end.
This concept was first presented by Yann le cun in 1998 for digit classification where he used a single convolution layer.

**CNN(Convolutional Neural Networks)**

CNN image classifications takes an input image, process it and classify it under certain categories (Eg., Dog, Cat, Tiger, Lion). Computers sees an input image as array of pixels and it depends on the image resolution.
Convolution is the first layer to extract features from an input image. Convolution preserves the relationship between pixels by learning image features using small squares of input data. It is a mathematical operation that takes two inputs such as image matrix and a filter or kernal.
Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters.

ReLU stands for Rectified Linear Unit for a non-linear operation. 

The output is **ƒ(x) = max(0,x).**

Why ReLU is important : ReLU&#39;s purpose is to introduce non-linearity in our ConvNet. Since, the real world data would want our ConvNet to learn would be non-negative linear values.
Pooling layers section would reduce the number of parameters when the images are too large
Max pooling take the largest element from the rectified feature map. Taking the largest element could also take the average pooling. Sum of all elements in the feature map call as sum pooling.
The layer we call as FC layer, we flattened our matrix into vector and feed it into a fully connected layer like neural network.

**Plain Network**

Our plain baselines  are mainly inspired by the philosophy of VGG nets . The convolutional layers mostly have 3×3 filters and
follow two simple design rules: 

(i) for the same output feature map size, the layers have the same number of filters; and 
(ii) if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.

We perform downsampling directly by convolutional layers that have a stride of 2. The network ends with a global average pooling layer and a 1000-way fully-connected layer with softmax. The total number of weighted layers is 34  (middle). It is worth noticing that our model has fewer filters and lower complexity than VGG nets. Our 34-layer baseline has 3.6 billion FLOPs (multiply-adds), which is only 18% of VGG-19 (19.6 billion FLOPs).

**Problem:**

When deeper networks starts converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated and then degrades rapidly.

**How to solve?**

Instead of learning a direct mapping of x -\&gt;y with a function H(x) (A few stacked non-linear layers). Let us define the residual function using F(x) = H(x) — x, which can be reframed into H(x) = F(x)+x, where F(x) and x represents the stacked non-linear layers and the identity function(input=output) respectively.

Residual block:
There are two kinds of residual connections:

  1.The identity shortcuts (x) can be directly used when the input and output are of the same dimensions.
 
  2. When the dimensions change,

   - The shortcut still performs identity mapping, with extra zero entries padded with the increased dimension.
   - The projection shortcut is used to match the dimension (done by 1\*1 conv) using the following formula(2).

**What is Transfer Learning?**

Transfer learning, is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.

**Why Transfer Learning?**

- In practice a very few people train a Convolution network from scratch (random initialisation) because it is rare to get enough dataset. So, using pre-trained network weights as initialisations or a fixed feature extractor helps in solving most of the problems in hand.
- Very Deep Networks are expensive to train. The most complex models take weeks to train using hundreds of machines equipped with expensive GPUs.

**How Transfer Learning helps ?**

When we look at what these Deep Learning networks learn, they try to detect edges in the earlier layers, Shapes in the middle layer and some high level data specific features in the later layers. These trained networks are generally helpful in solving other computer vision problems.

**Tools and Technology Used**

We used **Python** syntax for this project. As a framework we used **Keras** , which is a high-level neural network API written in Python. But Keras can&#39;t work by itself, it needs a backend for low-level operations. Thus, we installed a dedicated software library — Google&#39;s **TensorFlow**.

We have used Keras **Image Preprocessing** library for preprocessing the images of the dataset.

As a development environment we used the **Anaconda Distribution** and **Jupyter Notebook**.

We used **Matplotlib** for data visualization, **Numpy** for various array operations involved with the CNNs and **h5py** for saving the weights of the final model.

**Raw Data:**

The dataset we have used is a collection of identifiable professionals, in order to ensure that machine learning systems can be trained to recognize professionals by their mode of dressing as humans can observe. It contains 11,000 images that span cover 10 categories of professions. The professions included are:

- Chef
- Doctor
- Engineer
- Farmer
- Firefighter
- Judge
- Mechanic
- Pilot
- Police
- Waiter

There are 1,100 images for each category, with 900 images for trainings and 200 images for testing. The images contained in the dataset set are obtained from Google Images search and are arranged to respective category folder.


**Data Preprocessing:**

The images are first resized to 224 \* 224 pixel JPEG images. After that we have applied two techniques in order on our dataset to improve the accuracy.

- Normalisation : In order to achieve normalisation we have divided each pixel value by 255. The pixel range can now be described with a 0.0-1.0 where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF). Normalization helps to remove distortions caused by lights and shadows in an image.
- Data augmentation: We augment the existing data-set with perturbed versions of the existing images using real-time data augmentation provided in Keras Image Data Generator. This is done to expose the neural network to a wide variety of variations. This makes it less likely that the neural network recognizes unwanted characteristics in the data-set.

**Training Process:**

We train data on three two on Convolution Neural Networks (CNN) architecture. We use two variations of CNN - VGG-19 and ResNet50 to solve our problem. The loss function used is &quot;categorical\_crossentropy&quot; which is proven to be best loss function for image recognition problems.

**Model 1: VGG-19**

Different parameters of  this model are the following:

- Layers Used: 19 Conv2D layers, 5 MaxPool 2\*2 layers and 1 dense layer at the end.
- Batch\_size: 32
- Optimizer: Adam optimizer with initial learning rate 0.01 and decay 0.0001
- Total Trainable Parameters: 20,275,274

The training data is fed to this network and the model is trained for 53 epochs on the training data and validated by the validation data.



**Model 2: ResNet50**

Different parameters of  this model are the following:

- Layers Used:  34 Conv2D layers, 5 MaxPool layers and 1 dense layer at the end.
- Batch\_size: 32
- Optimizer: Adam optimizer with initial learning rate 0.01 and decay 0.0001
- Total Trainable Parameters: 23,555,082

The training data is fed to this network and the model is trained for 61 epochs on the training data and validated by the validation data.

**Testing Process:**

Model is tested on 2000 images having 200 images of the above mentioned 10 categories. To increase the diversity in dataset, we also used real time data augmentation on the test set and measure the accuracy of the models created.


**Result Analysis**

For our Project we have used two models.

The first model we have used is **VGG-19 convolutional neural network**. This model is trained over a dataset of 9,000 images. And the model is trained through the google co-lab notebook which took around 1 day for our data set. The model&#39;s testing accuracy is 72%.

The second model we have used is **RESNET-50**. This model is hybrid integration of Residual network integration and Deep architecture parsing. For our dataset this model have 79% of testing accuracy. Which is better than the first model.

Once we are done with the of training our model we are now ready to predict the image/profession identification through it. In our model We are using 10 classes of Profession hence our model will be based on the probability of finding the profession of any image we are using top 3 classes for any prediction. The model we have used give the result as follows.The Prediction for the image is given above.


**Conclusion**

We are identifying professions based on datasets available and how we trained them using VGG19 and RESNET 50 model using transfer learning. Though it didn&#39;t have much effect on time but it improved efficiency .

In practice a very few people train a Convolution network from scratch (random initialisation) because it is rare to get enough dataset. So, using pre-trained network weights as initialisations or a fixed feature extractor helps in solving most of the problems in hand.

When we look at what these Deep Learning networks learn, they try to detect edges in the earlier layers, Shapes in the middle layer and some high level data specific features in the later layers. These trained networks are generally helpful in solving other computer vision problems.
