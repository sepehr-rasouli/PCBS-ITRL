"""
Author: Sepehr Rasouli Jazi
"Information Theory: An Analysis of Supervised Learning"

In this project, we’ll analyze an example of supervised learning, specifically on the MNIST data set,
using information theory tools such as entropy and mutual information.
We’ll start by binarizing the data set and then plotting a heatmap of each digit summed over
the training data set.
Next, we’ll calculate the mutual information
between each pixel and its class label. Finally we’ll use the pixels with the highest mutual information
and calculate their prediction ability on the dataset.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
from sklearn.preprocessing import binarize
import seaborn as sns
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import minmax_scale


def plot_binarized_digit(img_vector, digit):
    # finding the square side length of a n*n array
    n = np.int(np.sqrt(img_vector.shape))
    fig = plt.figure()
    plt.imshow(img_vector.reshape(n, n), cmap='gray')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()
    fig.savefig('binarized-digit_' + str(digit) + '.png')


def plot_heatmap_binarized_digit(img_vector, digit):
    img_vector = np.array(img_vector)
    # finding the square side length of a n*n array
    n = np.int(np.sqrt(np.size(img_vector)))
    fig = plt.figure()
    sns.heatmap(img_vector.reshape(n, n))
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    fig.savefig('binarized-digit-heatmap_' + str(digit) + '.png')
    plt.show()


def entropy(vector_prob):
    """returns entropy of a probability mass function p ; H(x) = -∑ p * log2(p) """
    if vector_prob.ndim > 1:
        #         to account for 2d probability vector
        entropy = np.sum(-np.multiply(vector_prob, np.ma.log2(vector_prob)), axis=1)
        entropy = np.ma.fix_invalid(entropy, fill_value=0)
        return np.array(entropy)
    else:
        return -np.sum([np.multiply(p, np.ma.log2(p)) for p in vector_prob])


def top_n_pixels_mi(n):
    """returns top n pixels with the highest MI with data"""
    mnist_digits_number = 10
    top_pixel_max_MI_index = np.argpartition(MI_X_and_Y, -n)[-n:]
    top_pixels = bin_mnist_data[:, top_pixel_max_MI_index]
    counts_top_pixels = np.unique(top_pixels, axis=0, return_counts=True)[1]

    # H(X|Y) = ∑ p(y) * H(X|Y=y)
    entropy_X_given_Y = 0
    for digit in range(mnist_digits_number):
        p_X_given_Y = np.unique(top_pixels[bin_mnist_label == digit], axis=0, return_counts=True)[1] / \
                      bin_mnist_label[bin_mnist_label == digit].shape[0]
        entropy_X_given_Y += prob_Y[digit] * entropy(p_X_given_Y)

    prob_top_pixels = counts_top_pixels / bin_mnist_data.shape[0]
    # MI(X, Y) = H(X) - H(X|Y)
    top_n_MI_X_and_Y = entropy(prob_top_pixels) - entropy_X_given_Y

    return top_n_MI_X_and_Y


def prob_y_given_x(x):
    """returns an array for each digit calculating P(Y|X=x)"""
    mnist_img_vector_size = 784
    mnist_digits_number = 10
    prob_Y_given_X_equals_x = np.zeros((mnist_img_vector_size, mnist_digits_number))
    for pixel in range(mnist_img_vector_size):
        pixel_total_count_for_x = bin_mnist_data[bin_mnist_data[:, pixel] == x].shape[0]

        digit_count_in_labels = np.zeros(mnist_digits_number)
        pixel_equals_x_values, pixel_equals_x_count = np.unique(bin_mnist_label[bin_mnist_data[:, pixel] == x],
                                                                return_counts=True)
        if len(pixel_equals_x_values) > 0:
            if len(pixel_equals_x_values == 1):
                digit_count_in_labels[pixel_equals_x_values] = pixel_equals_x_count
            else:
                for digit, digit_count in pixel_equals_x_values, pixel_equals_x_count:
                    digit_count_in_labels[digit] = digit_count

        # To catch RuntimeWarning: invalid value encountered in true_divide
        if pixel_total_count_for_x > 0:
            probability = digit_count_in_labels / pixel_total_count_for_x
            prob_Y_given_X_equals_x[pixel] = probability

    return prob_Y_given_X_equals_x


def pixel_prediction(index, pixel):
    """"find true positive and true negatives prediction of a pixel in data"""
    if bin_mnist_data[index, pixel] == 1:
        return np.argmax(prob_Y_given_X_equal_1[pixel])
    # elif statement for readability instead of else
    elif bin_mnist_data[index, pixel] == 0:
        return np.argmax(prob_Y_given_X_equal_0[pixel])


def array_to_string(array):
    return ''.join(str(s) for s in array)


def prediction_top_n(n):
    """return the top n pixels with the best prediction accuracy, unique pixel patterns found and classifier """

    # top n pixel indexes
    top_pixels_index_list = np.argpartition(pixel_prediction_accuracy, -n)[-n:]
    top_pixels_dataset = bin_mnist_data[:, top_pixels_index_list]

    unique_patterns = np.unique(top_pixels_dataset, axis=0, return_counts=True)[0]

    # build a dictionary that predicts a class label according to the top-n pixels
    pred_dict = {}

    for pattern in unique_patterns:

        # pattern_found_bool_leas = (top_pixels_dataset == pattern).all(axis=1).nonzero()[0]
        # finds pattern's occurance and count in top_pixels_dataset
        digit_list, digit_counts = np.unique(bin_mnist_label[(top_pixels_dataset == pattern).all(axis=1).nonzero()[0]],
                                             return_counts=True)
        # calculate the probability of the pattern present in labels
        prob_pattern_present = digit_counts / bin_mnist_label[(top_pixels_dataset == pattern).all(axis=1).nonzero()[0]].shape[0]

        predictor = digit_list[np.argmax(prob_pattern_present)].astype(np.int)
        pred_dict[array_to_string(pattern)] = predictor

    return top_pixels_index_list, unique_patterns, pred_dict


mnist_digits_number = 10
mnist_img_vector_size = 784
#     finding the side length of a digit's square image
sq_len = np.int(np.sqrt(mnist_img_vector_size))

mnist = fetch_openml('mnist_784', cache=False)

bin_mnist_data = binarize(mnist.data)
bin_mnist_label = mnist.target.astype(np.int)

# reduce  if the run is too slow(for optimal results to 6000), note that reducing it to less than 3000,
# would result in drastically lower maximum accuracy and different results of I(x,y), H(x) and slightly different plots

database_size = 10000

bin_mnist_test, bin_mnist_test_label = bin_mnist_data[database_size:2 * database_size], bin_mnist_label[database_size:2 * database_size]
bin_mnist_data, bin_mnist_label = bin_mnist_data[0:database_size], bin_mnist_label[0:database_size]

digit_samples = []
for i in range(mnist_digits_number):
    digit_samples.append(bin_mnist_data[np.where(bin_mnist_label == i)])


# adding all vectors of one digit's values to make one digit heatmap
digits_sum_list = []
for j in range(mnist_digits_number):
    sum = 0
    for i in digit_samples[j]:
        sum += i
    digits_sum_list.append(sum)

digits_sum_list = np.array(digits_sum_list)
# normalize array
digits_sum_list = digits_sum_list / np.linalg.norm(digits_sum_list)


# plot 80th batch of samples
for i in range(mnist_digits_number):
    plot_binarized_digit(digit_samples[i][80], i)

for i in range(10):
    plot_heatmap_binarized_digit(digits_sum_list[i], i)


# calculate the probability of class lablels for each digit
prob_Y = np.unique(bin_mnist_label, return_counts=True)[1] / bin_mnist_label.shape[0]

print('Entropy of Labels =', entropy(prob_Y))

prob_Y_given_X_equal_0 = prob_y_given_x(x=0)
prob_Y_given_X_equal_1 = prob_y_given_x(x=1)

prob_X_equals_1 = bin_mnist_data.mean(axis=0)
prob_X_equals_0 = 1 - bin_mnist_data.mean(axis=0)

# mutual_information(X,Y), I(X|Y) = H(Y) – H(Y|X=0) * P(X=0) - H(Y|X=1) * P(X=1)
MI_X_and_Y = entropy(prob_Y) - entropy(prob_Y_given_X_equal_0) * prob_X_equals_0 - entropy(
    prob_Y_given_X_equal_1) * prob_X_equals_1

print('MI(X,Y): min={}, max={}, mean={}, median={}'.format(round(MI_X_and_Y.min(),3), round(MI_X_and_Y.max(),3), round(MI_X_and_Y.mean(),3), round(np.median(MI_X_and_Y),3)))

fig = plt.figure()
sns.heatmap(MI_X_and_Y.reshape(sq_len, sq_len))
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("pixel mutual information with the class label")
plt.savefig("pixel-mutual-information.png")
plt.show()

# Ranking pixels by prediction accuracy

pixel_prediction_accuracy = np.zeros(mnist_img_vector_size)
# find high MI pixels which predict the label right
for pixel in range(mnist_img_vector_size):
    correct = 0
    for i in range(database_size):
        prediction = pixel_prediction(i, pixel)
        label = bin_mnist_label[i]

        if prediction == label:
            correct += 1

    pixel_prediction_accuracy[pixel] = correct / database_size

fig = plt.figure()
sns.heatmap(pixel_prediction_accuracy.reshape(sq_len, sq_len))
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Prediction accuracy of each pixel on the training data")
plt.savefig("pixel-accuracy.png")
plt.show()


# Normalizing the prediction accuracy heatmap to minimun and maximum of the MI pixel accuracy

normalized_pixel_prediction_accuracy=minmax_scale(pixel_prediction_accuracy, (np.min(MI_X_and_Y), np.max(MI_X_and_Y)))

# This creates a non-normalizd diff for comparison
sns.heatmap((MI_X_and_Y - normalized_pixel_prediction_accuracy).reshape(sq_len, sq_len))
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("MI differentiated by normalized pixel accuracy")
plt.savefig("MI-differentiated-by-normalized-pixel-accuracy.png")
plt.show()

sns.heatmap((MI_X_and_Y - pixel_prediction_accuracy).reshape(sq_len, sq_len))
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("MI differentiated by pixel accuracy")
plt.savefig("MI-differentiated-by-pixel-accuracy.png")
plt.show()

# range of number of top high MI pixels selected
n_range = list(range(1, 50)) + list(range(50, 90, 10))
top_n_pixels_MI_list = []
for n in n_range:
    top_n_pixels_MI_list.append(top_n_pixels_mi(n))

# create smooth line
xnew = np.linspace(np.min(n_range), np.max(n_range), 200)
spl = make_interp_spline(n_range, top_n_pixels_MI_list, k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth, color='r', alpha=0.5)

plt.scatter(n_range, top_n_pixels_MI_list, s=20)
plt.grid(True)
plt.set_xticks = n_range
plt.xlabel('number of pixels selected')
plt.ylabel('Mutual information with class label')
plt.title('The top n-pixels mutual information with the class label')
plt.savefig('top-n-mi.png')
plt.show()


# reduce the range of n-values if the run is too slow
n_values = list(range(1, 10)) + list(range(10, 85, 5))

# find accuracy by predicting with prediction dictionary on the test dataset
accuracy_list = []
for n in n_values:
    print("n=", n)

    top_pixel_index, unique_patterns, pred_dict = prediction_top_n(n)

    found_count = 0

    for i in range(bin_mnist_test_label.shape[0]):
        pattern_found = bin_mnist_test[i, top_pixel_index]
        if pattern_found is not None:
            pattern_string = array_to_string(pattern_found)
            if pred_dict.get(pattern_string) == bin_mnist_test_label[i]:
                found_count += 1

    accuracy = found_count / bin_mnist_test_label.shape[0]
    print("Accuracy = ", round(accuracy, 3), '\n')
    accuracy_list.append(accuracy)

print("Maximum Accuracy Achieved= ", round(np.max(accuracy_list), 3))

# create smooth line
xnew = np.linspace(np.min(n_values), np.max(n_values), 200)
spl = make_interp_spline(n_values, accuracy_list, k=1)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth, color='r')

plt.scatter(n_values, accuracy_list, s=20)
plt.grid(True)
plt.set_xticks = n_values
plt.title('prediction accuracy of top-n high MI pixels')
plt.xlabel('number of top-n high MI pixels selected')
plt.ylabel('accuracy')
plt.savefig('accuracy-n-high-MI-pixels.png')
plt.show()
