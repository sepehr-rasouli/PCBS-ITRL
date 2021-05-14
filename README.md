**Package Requirements:**

Python v3, numpy, scipy, sklearn, matplotlib, seaborn

**"****Information Theory: An Analysis of Supervised Learning"**

In this project, we'll analyze an example of supervised learning,
specifically the MNIST data set, using information theory tools such as
entropy and mutual information. We'll start by binarizing the data set
and then plotting a heatmap of each digit summed over the training data
set. Next, we'll calculate the mutual information between each pixel and
its class label. Finally, we'll use the pixels with the highest mutual
information and calculate their prediction ability on the dataset.

1.  **"****Importing the data set, binarizing and plotting the digits"**

    In this project we'll use a binarized version of the MNIST dataset,
    which is a large database of handwritten digits that is commonly
    used for training various image processing systems.\
    \
    we'll import the dataset and binarize it using sklearn's function:

    mnist = fetch\_openml(\'mnist\_784\', cache=False)

    binary\_mnist\_data = binarize(mnist.data)

    binary\_mnist\_target = mnist.target.astype(np.int)

    Now, we have both the training set and label set.

    Each digit has a vector of size 784 and we'll reshape it to a 28\*28
    square to plot the digits. Labels are digits from 0 to 9.\
    We'll select 10,000 images from the training set and another 10,000
    from the test set.

    By using the plot\_binarized\_digit function we'll plot the 10^th^
    batch of the training set for a demonstration of binarization:

    ![](media/image1.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image2.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image3.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image4.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image5.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image6.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image7.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image8.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image9.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}![](media/image10.png){width="1.1811023622047243in"
    height="0.7874015748031497in"}\
    By adding up all vectors of each digit we'll make a digit pixel
    heatmap to see which pixels most represent the digit:

    ![](media/image11.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image12.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image13.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image14.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image15.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image16.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image17.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image18.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image19.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}![](media/image20.png){width="2.3622047244094486in"
    height="1.5748031496062993in"}

    As expected, the heatmap represents the optimal shape of each digit.

2.  **"****Information Theory Tools Analysis"**

    We'll start by calculating the probability distribution of the class
    labels:

    p\_Y = np.unique(binary\_mnist\_target, return\_counts=True)\[1\] /
    binary\_mnist\_target.shape\[0\]

    And we'll calculate the entropy of probability mass function p ,H(x)
    = -∑ p \* log2(p) :

    def entropy(vector\_prob):

    if vector\_prob.ndim \> 1:

    \# to account for 2d probability vector

    entropy = np.sum(-np.multiply(vector\_prob,
    np.ma.log2(vector\_prob)), axis=1)

    entropy = np.ma.fix\_invalid(entropy, fill\_value=0)

    return np.array(entropy)

    else:

    return -np.sum(\[np.multiply(p, np.ma.log2(p)) for p in
    vector\_prob\])

    The entropy for the class labels is 3.318 bits being close to the
    entropy of a uniform distribution , which is log~2~ (10) = 3.321.

    Now, we could estimate the mutual information of each pixel and the
    class label. This would result in knowing which pixels represent the
    class label better. The mutual information, have been implemented by
    the following formula, I(X,Y) = H(Y) -- H(Y\|X).

    Each pixel, either represent its class label or not, thus we
    calculate P(Y\|X=0) and P(Y\|X=1) .

    The function P\_Y\_given\_X calculates this probability by counting
    the times each pixel has correctly predicted the label for all the
    digits.\
    Finally, the mutual information is calculated by :\
    I(X\|Y) = H(Y) -- H(Y\|X=0) \* P(X=0) - H(Y\|X=1) \* P(X=1)

    For I(X,Y), min=0, max=0.336, mean=0.089, and median=0.06.\
    This shows some pixels have no mutual information with the class
    while the best ones have a mutual information of 0.336.

    The plot for the mutual information of each digit and the labels is
    the following figure:

![](media/image21.png){width="4.3203882327209095in"
height="2.880259186351706in"}

Also, the prediction accuracy of each pixel on the training data is
calculated by choosing each pixel and looking for its correct prediction
(both true positives and true negatives) on the target set divided by
the total number of the images in the selected data set. The pixels with
high prediction accuracy have a high concentration in the middle of the
image due to the difference between different digits in this area. The
plot is the following figure:\
![](media/image22.png){width="4.148648293963254in"
height="2.7657655293088363in"}

To compare pixel accuracy and pixel mutual information, we differentiate
the pixel accuracy plot and the mutual information plot (both normalized
and not normalized). We could see how well mutual information represents
prediction accuracy since the two plots are nearly identical with small
areas of difference. As seen in the next plot there are only some areas
in the top center and left bottom corner that have a high MI but low
accuracy. However, it needs to be mentioned that they are still pixels
with low mutual information that could predict the labels well, and
mutual information is only one parameter for prediction accuracy of
pixels. Later on, we'll discuss this in full detail by only choosing the
pixels with highest mutual information and predicting the image only
using them.

![](media/image23.png){width="3.830097331583552in"
height="2.553397856517935in"}

![](media/image24.png){width="3.829861111111111in"
height="2.5532403762029747in"}

3.  **"Prediction Power of Mutual Information in Supervised Learning"**

    In the previous section, the mutual information of pixels and the
    pixel accuracy has been analyzed. In this section, we'll try to
    directly test the predictive power of pixels with high mutual
    information on the labels using the test data.

    First, we'll descendingly sort the pixels based on their mutual
    information with the class label. Next, we'll select the top n
    pixels in the range of 1 to 80 and calculate each batch of selected
    pixels' mutual information(joint distribution of them) with the
    class labels. The plot is the following:

![](media/image25.png){width="3.6666666666666665in"
height="2.4444444444444446in"}

According to the plot, after 30 pixels the maximum entropy is reached
which is almost equal to the entropy of the class label.

Finally. we'll only use the top-n pixels with the highest mutual
information with the class in the previous part, to predict the class
labels. Since these top-n pixels have a high mutual information, they
could provide a pattern to classify the images.

For a range of n from 1 to 80, we'll construct a classifier which
consists only these n pixels. Then, we use this classifier on the test
data set and if we find a pattern including only these pixels, we have
successfully predicted the label. The plot of the prediction accuracy
for the forementioned n pixels is the following:

![](media/image26.png){width="3.846153762029746in"
height="2.5641021434820646in"}

As seen in the plot accuracy increases from the single pixel accuracy at
0.21 until we reach a maximum accuracy of 0.525 and then it drops to
zero, significantly below the single pixel accuracy. The analysis is
that there is a limit to finding patterns with high mutual information
that fully predict the labels. This is because there are pixels with low
mutual information which are used by the network, therefore not present
in our selected pixels. While the maximum prediction accuracy is not
considered a high value, it still shows the significant prediction value
of mutual information between data and labels.

In conclusion, mutual information is a good tool to understand how
neural networks work under the hood. This is a starting point to further
discover what the neural network is doing to predict the data. Further,
using information theory tools to evaluate the performance of deep
neural networks is explored in the \"Opening the Black Box of Deep
Neural Networks via Information\" paper by Tishby. He attempts to build
a comprehensive theory, namely information bottle neck theory, to find
out the relation between mutual information and DNNs. Briefly, the
theory is about how compressing the useful information in the data is
represented by the mutual information between the data and the middle
layers of the network.

\#Remarks for PCBS:

> The project has been coded in a Jupyter notebook for easier debugging
> due to the long run time of each section. But I have finally decided
> to put all of my code into a singly python file to conform to the
> rules and to have the same structure as the other students. The
> README.MD will walk you through the project step by step. While the
> python file creates the results and figures at once, all of them will
> be saved in the same directory.
>
> \# References

1.  Studied the workflow of training a network on MNIST in the following
    website:

> <https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/>

2.  A cool visualized neural net for handwritten digit recognition in
    Javascript that I experimented with.\
    <http://myselph.de/neuralNet.html>

3.  Studied basic information theory concepts and information bottle
    neck from Tishby's papers \"Opening the Black Box of Deep Neural
    Networks via Information\" and "The information bottleneck".\
    \
    \[1\] Tishby, N., Pereira, F. C., & Bialek, W. (2000). The
    information bottleneck method. *arXiv preprint physics/0004057*.

    \[2\] Shwartz-Ziv, R., & Tishby, N. (2017). Opening the black box of
    deep neural networks via information. *arXiv preprint
    arXiv:1703.00810*.

4.  PEP 8 \-- Style Guide for Python Code\
    <https://www.python.org/dev/peps/pep-0008/>

    \#TODO

<!-- -->

1.  The probability functions like the conditional
    probability(P\_Y\_given\_x) are not reusable since they only work in
    the dimensions of this project.

2.  Figures could be done by plotly which offers better tools for
    analysis.

3.  The figures are all in the same path of the main project while they
    should be saved in dedicated folders. This is due to various
    combability issues with how Jupyter Notebook and python3 handle
    files and issues with running the project on another device for
    testing.

4.  The entropy function runtime could be improved by using nump/math
    entropy function which seems to have the fastest runtime (according
    to <https://stackoverflow.com/a/45091961>)

5.  By using all the MNIST images (limited to 10,000 for this project
    because of the limited processing power available due to using an
    old processor), we would be able to acquire more accurate results.

6.  This project could be extended to include the information bottle
    neck methods

7.  The readability could be further improved by .
