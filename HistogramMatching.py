# Importing the libraries
import numpy as np
import os 
import cv2 # used only for loading the image
import matplotlib.pyplot as plt # used only for displaying the image
from google.colab import files # import file operations

class HistogramMatching:
    def calculate_histogram(self, image, num_bins=256):
        histogram = np.zeros(num_bins, dtype=np.int32) # initialize the histogram
        for pixel_value in image.ravel(): # ravel() returns a contiguous flattened array
            histogram[pixel_value] += 1 # increment the count of the pixel value
        return histogram # return the histogram

    # this function is responsible for calculating the normalized histogram of an image
    def calculate_normalized_histogram(self, image, num_bins=256):
        histogram = self.calculate_histogram(image, num_bins) # calculate the histogram
        sum_of_histogram = np.sum(histogram) # sum of all the pixel values
        histogram = histogram / sum_of_histogram # normalize the histogram
        return histogram # return the normalized histogram

    # this function is responsible for calculating the cumulative histogram of an image
    def calculate_cumulative_histogram(self, histogram):
        sum_of_histogram = np.sum(histogram) # sum of all the pixel values
        histogram = histogram / sum_of_histogram # normalize the histogram
        cumulative_histogram = np.zeros(histogram.shape, dtype=np.float32) # initialize the cumulative histogram
        cumulative_histogram[0] = histogram[0] 
        for i in range(1, histogram.shape[0]): # calculate the cumulative histogram
            cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]
        return cumulative_histogram # return the cumulative histogram

    # this function is responsible for matching the histogram of an image to the histogram of a reference image
    def match_histograms(self, image, reference_image):
        mapping = self.get_mapping(image, reference_image) # get the mapping
        matched_image = np.zeros(image.shape, dtype=np.uint8) # initialize the matched image
        for i in range(image.shape[0]): # match the image
            for j in range(image.shape[1]):
                matched_image[i, j] = mapping[image[i, j]]
        return matched_image # return the matched image

    # this function is responsible for matching the histogram of an image to the histogram of a reference image
    def get_mapping(self, image, reference_image, gray_levels=256):
        histogram = self.calculate_histogram(image) # calculate the histogram of the image
        cumulative_histogram = self.calculate_cumulative_histogram(histogram) # calculate the cumulative histogram of the image
        reference_histogram = self.calculate_histogram(reference_image) # calculate the histogram of the reference image
        reference_cumulative_histogram = self.calculate_cumulative_histogram(reference_histogram) # calculate the cumulative histogram of the reference image

        mapping = np.zeros(gray_levels) # initialize the mapping
        for pixel_value in range(gray_levels):
            old_value = cumulative_histogram[pixel_value] # get the cumulative histogram of the image
            temp = reference_cumulative_histogram - old_value # get the difference between the cumulative histogram of the reference image and the cumulative histogram of the image
            new_value = np.argmin(np.abs(temp)) # get the index of the minimum value in the difference
            mapping[pixel_value] = new_value # map the pixel value to the new value
        return mapping # return the mapping

    def run(self):
        # Upload the Source Image
        print('Upload the low quality image')
        source = files.upload()
        source_filename = list(source.keys())[0]
        print(f'File "{source_filename}" uploaded successfully!')
        print('----------------------------------------------------------------')
        # Upload the Target Image
        print('Upload the high quality image')
        target = files.upload()
        target_filename = list(target.keys())[0]
        print(f'File "{target_filename}" uploaded successfully!')
        print('----------------------------------------------------------------')
        print('Performing Matching')
        # Perform Matching    
        source_image = cv2.imread('/content/'+source_filename)
        target_image = cv2.imread('/content/'+target_filename) 
        matching = self.match_histograms(source_image, target_image)  
        # Download the result
        result = cv2.cvtColor(matching, cv2.COLOR_BGR2RGB)
        cv2.imwrite('matched_image_A.jpg', result)
        cv2.imwrite('matched_image_B.jpg', matching)
        files.download('matched_image_A.jpg')
        files.download('matched_image_B.jpg')
