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

    # performing histogram matching and displaying the images and their histograms
    def histogram_matching_and_visualization(self, image, reference_image, visualize=True): 
        matched_image = self.match_histograms(image, reference_image) # match the histogram of the image to the histogram of the reference image
        image_set = [image, reference_image, matched_image] 
        image_titles = ['Source Image', 'Target Image', 'Matched Image']
        if visualize:
            self.visualize_histograms(image_set, image_titles=image_titles) # display the images and their histograms   
        return matched_image # return the matched image  

    # this function is responsible for displaying the images and their histograms
    def visualize_histograms(self, image_set, figsize=(15, 9), image_titles=None):
        plt.figure(figsize=figsize) # set the figure size
    
        for i, image in enumerate(image_set): # iterate over all the images
            histogram = calculate_histogram(image) # calculate the histogram of the image
            normalized_histogram = calculate_normalized_histogram(image) # calculate the normalized histogram of the image
            cumulative_histogram = calculate_cumulative_histogram(histogram) # calculate the cumulative histogram of the image
            
            plt.subplot(3, len(image_set), i + 1) # display the image
            plt.imshow(image, cmap='gray')
            plt.title(image_titles[i], fontsize=10)
            plt.axis('off')
    
            plt.subplot(3, len(image_set), i + 1 + len(image_set)) # display the histogram
            plt.bar(range(256), normalized_histogram, width=1.0)
            plt.title('Normalized Histogram ('+image_titles[i]+')', fontsize=8)
            plt.xlabel('Pixel Value', fontsize=8)
            plt.ylabel('Frequency', fontsize=8)
    
            plt.subplot(3, len(image_set), i + 1 + 2 * len(image_set)) # display the cumulative histogram
            plt.plot(cumulative_histogram)
            plt.title('Cumulative Histogram ('+image_titles[i]+')', fontsize=8)
            plt.xlabel('Pixel Value', fontsize=8)
            plt.ylabel('Frequency', fontsize=8)
    
        plt.tight_layout()
        plt.show()
    
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
        matching = self.histogram_matching_and_visualization(source_image, target_image)  
        # Download the result
        cv2.imwrite('matched_image.jpg', matching)
        files.download('matched_image.jpg')

class HistogramMatchingCV2:
    def hist_match(self, source, template):
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)
        template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2Lab)
        source_hist, _ = np.histogram(source_lab[:, :, 0], bins=256, range=(0, 256))
        template_hist, _ = np.histogram(template_lab[:, :, 0], bins=256, range=(0, 256))
        source_cdf = source_hist.cumsum() / source_hist.sum()
        template_cdf = template_hist.cumsum() / template_hist.sum()
        lookup_table = np.interp(source_cdf, template_cdf, np.arange(256))
        matched_l = np.interp(source_lab[:, :, 0], np.arange(256), lookup_table)
        matched_lab = np.stack((matched_l.astype(np.uint8), source_lab[:, :, 1], source_lab[:, :, 2]), axis=-1)
        matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_Lab2BGR)
        return matched_bgr

    def perform_histogram_matching(self, dull_image_path, sharp_image_path):
        dull_image = cv2.imread(dull_image_path)
        sharp_image = cv2.imread(sharp_image_path)
        matched_image = self.hist_match(dull_image, sharp_image)
        return matched_image

    def run(self):
        # Upload the Source Image
        print('Upload the low quality image')
        source = files.upload()
        source_filename = list(source.keys())[0]
        print(f'File "{source_filename}" uploaded successfully!')

        # Upload the Target Image
        print('Upload the high quality image')
        target = files.upload()
        target_filename = list(target.keys())[0]
        print(f'File "{target_filename}" uploaded successfully!')

        result = self.perform_histogram_matching(dull_image_path='/content/'+source_filename, sharp_image_path='/content/'+target_filename)
        cv2.imwrite('matched_image.png', result)
        files.download('matched_image.png')
