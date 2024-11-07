import sys
import os

import image
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct


from collections import OrderedDict
import pywt


class Translator():
  

###############__EXERCISE 2__#####################
    def __init__(self): #The __init__ method is a constructor that is called each time an instance of the class is created.
        #self represents the instance of the class being created.
        pass

        
    def rgb_to_yuv(self, R, G, B): # https://www.cs.sfu.ca/mmbook/programming_assignments/additional_notes/rgb_yuv_note/RGB-YUV.pdf
        conversion_matrix = np.array([[ 0.299,  0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615,  -0.51499, -0.10001]])
        
        rgb = np.array([R, G, B])
        return np.matmul(conversion_matrix, rgb)

    def yuv_to_rgb(self, Y, U, V): # https://www.cs.sfu.ca/mmbook/programming_assignments/additional_notes/rgb_yuv_note/RGB-YUV.pdf
        conversion_matrix = np.array([[ 1,  0, 1.13983],
                         [1, -0.39465, -0.58060],
                         [1,  2.03211, 0]])
        
        yuv = np.array([Y, U, V])
        return np.matmul(conversion_matrix, yuv) 
    
###############__EXERCISE 3__##################### 

    def resize(self, width, height, input_image, output_image): #https://trac.ffmpeg.org/wiki/Scaling
        command = [
            'ffmpeg',
            '-i', input_image,
            '-vf', f'scale={width}:{height}',
             '-frames:v', '1',
            output_image
            ]
        
        subprocess.run(command, check=True)


###############__EXERCISE 5__#####################

    def black_and_white(self, input_image, output_image): 
        command = [
            'ffmpeg',
            '-i', input_image,
            '-vf', 'format=gray',
             '-qscale:v', '31', output_image
            ]
       
        
        subprocess.run(command, check=True)

# INFORMATION ON THE FUNCTION AND WHAT THE PARAMETERS DO:

# For JPEG/JPG, -qscale controls image quality. The valid range is 2 to 31. Lower value is higher quality value. 
# Value 0 means lossless encoding. Value 1 is not recommended because of much larger file, little quality difference,
# and may causing problems on decoding and encoding. Thus  highest compression is achieved at 31, which is noticeably lossy.
# Source: http://www.output.to/sideway/default.aspx?qno=200101402

 # For the format filter: https://stackoverflow.com/questions/32384057/ffmpeg-black-and-white-conversion

 # format=gray:
 #  1. The filter converts the image to a grayscale format by only keeping the luminance component, 
 #     effectively making all color pixels shades of gray based on their brightness levels.

#   2. It changes the pixel format to a single-channel format, using 8 bits per pixel, 
#      where each pixel’s value corresponds to a brightness level from black (0) to white (255).
#      Removing these color channels significantly reduces the amount of data, increasing compression.


# COMMENT THE VISUAL RESULTS:

# The original color image of Snoop Dogg, at 94 KB, undergoes significant changes when converted to grayscale and compressed with a high compression setting (qscale 31). 
# The grayscale conversion, using the format=gray filter, simplifies the image by retaining only the luminance channel, discarding chrominance (color) information. 
# This reduces the image to shades of gray based on brightness levels, removing any color complexity and making the file easier to compress. When the image is compressed 
# with qscale 31, the file size drops drastically to just 16 KB. This aggressive compression discards a considerable amount of image detail, especially fine textures and 
# high-frequency components. JPEG compression works by first converting the image to the YCbCr color space, then applying Discrete Cosine Transform (DCT) to break the 
# image into blocks. High-frequency details (edges) are retained in the DCT coefficients, but many are discarded during quantization at high compression levels, 
# leading to blurring and noticeable artifacts. The result is a blurry image where sharp details, such as the eyes, are less distinct, though edges remain somewhat recognizable. 
# In contrast, areas with flat colors, like the background, are better maintained since they don’t have complex textures or variations, so the compression is less noticeable. 
# Basically we could say that the final image is still recognizable but noticeably degraded.


    def run_length_encode(self, byte_sequence):
            # Generate ordered dictionary of all lower
            # case alphabets. For this input "wwwwaaadexxxxxx" its output will be
            # dict = {'w':0, 'a':0, 'd':0, 'e':0, 'x':0}
        dict=OrderedDict.fromkeys(byte_sequence, 0)

            # Now iterate through input string to calculate
            # frequency of each character, its output will be
            # dict = {'w':4,'a':3,'d':1,'e':1,'x':6}
        for ch in byte_sequence:
            dict[ch] += 1

            # now iterate through dictionary to make
            # output string from (key,value) pairs
        output = ''
        for key,value in dict.items():
            output += str(key) + str(value)  
        #For each character (key) and its count (value), we add them to the output string. so that it contains
        #each character and after the number of times it appears as a string "w4a3d1e1x6"
        
    
        return output
     
#CODING THE FUNCTION: https://www.geeksforgeeks.org/run-length-encoding-python/
# As we have seen in theory, RLE is used in the encoding part of the JPEG block diagram (RLE-Huffman encoding). 
# Where RLE is used to encode runs of zeros following quantization, especially in higher-frequency components.

###############__EXERCISE 6__#####################
class DCT:
    # The family of DCTs (DCT type- 1,2,3,4) is the outcome of different combinations of homogeneous boundary conditions. 
    # DCT type-II can be implemented using Fast fourier transform (FFT) internally, as the transform is applied on real values, 
    # Real FFT can be used. DCT4 is implemented using DCT2 as their implementations are similar except with some added pre-processing 
    # and post-processing. DCT2 implementation can be described in the following steps:
    #    1. Re-ordering input
    #    2. Calculating Real FFT
    #    3. Multiplication of weights and Real FFT output and getting real part from the product.
    #
    #  https://arm-software.github.io/CMSIS-DSP/v1.10.1/group__DCT4__IDCT4.html

    def __init__(self, type_of_dct):
         
         self.type_of_dct = int(type_of_dct)

    def encode(self, input_data):
        #https://www.geeksforgeeks.org/python-scipy-fft-dct-method/
        return dct(input_data, type=self.type_of_dct)

    def decode(self, dct_data):
        #https://www.geeksforgeeks.org/python-scipy-fft-idct-method/
        return idct(dct_data, type=self.type_of_dct, norm='ortho')

    def visualize(self, original_data, encoded_data, decoded_data):
        """Visualize the original, encoded, and decoded data with labeled axes."""
        plt.figure(figsize=(12, 6))

        # Plot the original signal
        plt.subplot(1, 3, 1)
        plt.plot(original_data)
        plt.title("Original Data")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")

        # Plot the encoded (DCT) signal

        plt.subplot(1, 3, 2)
        plt.plot(encoded_data)
        plt.title(f"Encoded (DCT-{self.type_of_dct})")
        plt.xlabel("Coefficient Index")
        plt.ylabel("Magnitude")

        # Plot the decoded signal

        plt.subplot(1, 3, 3)
        plt.plot(decoded_data)
        plt.title("Decoded (Inverse DCT)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")


        plt.tight_layout()
        plt.show()

###############__EXERCISE 7__#####################

#CODING THE CLASS: https://pywavelets.readthedocs.io/en/latest/

class DWT:
    def __init__(self, wavelet_type):
        self.wavelet_type = wavelet_type

    def encode(self, input_data):
        # Apply the DWT using the specified wavelet type
        coeffs = pywt.dwt(input_data, self.wavelet_type)
        approximation_coeffs, detail_coeffs = coeffs
        return coeffs, detail_coeffs, approximation_coeffs

    def decode(self, coeffs):
        return pywt.idwt(coeffs[0], coeffs[1], self.wavelet_type)

    def visualize(self, input_data, encoded_data, approximation_coeffs, detail_coeffs, decoded_data):
        plt.figure(figsize=(12, 6))

        # Plot the original signal
        plt.subplot(2, 3, 1)
        plt.plot(input_data)
        plt.title("Original Data")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")

        # Plot the encoded (wavelet coefficients) signal
        plt.subplot(2, 3, 2)
        plt.plot(encoded_data[0])  # Plot the approximation coefficients
        plt.title(f"Encoded (DWT - {self.wavelet_type})")
        plt.xlabel("Coefficient Index")
        plt.ylabel("Magnitude")

        # Plot approximation coefficients (low-frequency components)
        plt.subplot(2, 3, 3)
        plt.plot(approximation_coeffs)
        plt.title('Approximation Coefficients (Low-Frequency)')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Magnitude')

        # Plot detail coefficients (high-frequency components)
        plt.subplot(2, 3, 4)
        plt.plot(detail_coeffs)
        plt.title('Detail Coefficients (High-Frequency)')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Magnitude')

        # Plot the decoded signal
        plt.subplot(2, 3, 5)
        plt.plot(decoded_data)
        plt.title("Decoded (Inverse DWT)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()


def main():
    while True:
        print("\nWelcome to the Interactive Image and Signal Processing Tool!")
        print("------------------------------------------------------------")
        print("1. RGB to YUV Conversion and Image Resizing")
        print("2. Black & White Image Converter with Hard Compression")
        print("3. Run-Length Encoding Tool")
        print("4. Exercise 4 (Function Empty)")
        print("5. DCT Encoding and Visualization Tool")
        print("6. DWT Encoding and Visualization Tool")
        print("7. Exit")

        choice = input("Please select an option (1-7): ").strip()

        if choice == '1':
            rgb_yuv_resizer()
        elif choice == '2':
            black_white_converter()
        elif choice == '3':
            run_length_encoding()
        elif choice == '4':
            print("Exercise 4 is under construction. Please choose another option.")
        elif choice == '5':
            dct_tool()
        elif choice == '6':
            dwt_tool()
        elif choice == '7':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option (1-7).")

def rgb_yuv_resizer():
    print("\nWelcome to the RGB-YUV Converter and Image Resizer!")
    print("----------------------------------------------------\n")

    print("Task 1: RGB to YUV Conversion")
    def get_rgb_value(channel):
        while True:
            try:
                value = int(input(f"Enter the {channel} (0-255): "))
                if 0 <= value <= 255:
                    return value
                else:
                    print("Invalid input! Please enter an integer between 0 and 255.")
            except ValueError:
                print("Invalid input! Please enter an integer between 0 and 255.")
    
    # Get valid RGB values
    R = get_rgb_value("R (Red)")
    G = get_rgb_value("G (Green)")
    B = get_rgb_value("B (Blue)")
    
    # Perform the RGB to YUV conversion
    translator = Translator()
    YUV = translator.rgb_to_yuv(R, G, B)
    print(f"\nConverted RGB({R}, {G}, {B}) to YUV: {YUV}")

    print("\nTask 2: YUV to RGB Conversion")
    convert_back = input("Do you want to convert back to RGB (y/n)? ").strip().lower()

    if convert_back == 'y':
        # Get valid YUV values from the user
        Y = float(input("Enter the Y value: ")) 
        U = float(input("Enter the U value: ")) 
        V = float(input("Enter the V value: "))

        # Perform the YUV to RGB conversion
        RGB = translator.yuv_to_rgb(Y, U, V)
        print(f"Converted YUV({Y}, {U}, {V}) back to RGB: {RGB}")

    print("\nTask 3: Image Resizing")
    resize_image = input("Do you want to resize an image (y/n)? ").strip().lower()

    if resize_image == 'y':
        while True:  
            try:
                input_image = input("Enter the path of the input image: ").strip()

                if not os.path.exists(input_image):
                    print(f"The path '{input_image}' does not exist. Please enter a valid path.")
                    continue  

                output_image = input("Enter the full path where you want to save the resized image (including filename and format, e.g., 'output/image_resized.jpeg'): ").strip()

                if not output_image.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.gif')):
                    print("Please provide a valid image format in the output filename (e.g., .jpeg, .png).")
                    continue  

                output_dir = os.path.dirname(output_image)
                if output_dir and not os.path.exists(output_dir):
                    print(f"The directory '{output_dir}' does not exist. Please enter a valid directory.")
                    continue  

                width = int(input("Enter the desired width for the image: "))
                height = int(input("Enter the desired height for the image: "))

                translator.resize(width, height, input_image, output_image)
                print(f"Image resized successfully! Output saved to: {output_image}")
                break  

            except subprocess.CalledProcessError as e:
                print(f"Error resizing image: {e}. Please try again.")
            except ValueError:
                print("Invalid input! Please enter integers for width and height.")
            except FileNotFoundError:
                print("The input image path was not found. Please try again with a valid path.")

    print("\nProcess complete. Thank you for using the RGB-YUV Converter and Image Resizer!")

def black_white_converter():
    print("\nWelcome to the Black & White Image Converter with Hard Compression!")
    print("----------------------------------------------------\n")

    print("Task 1: Convert Image to Black & White with Hard Compression")
    try:
        input_image = input("Enter the path of the input image (e.g., 'image.jpeg'): ").strip()

        if not os.path.exists(input_image):
            print(f"The path '{input_image}' does not exist. Please enter a valid path.")
            return

        output_image = input("Enter the full path where you want to save the black & white image (e.g., 'output/image_BW.jpeg'): ").strip()

        if not output_image.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.gif')):
            print("Please provide a valid image format in the output filename (e.g., .jpeg, .png).")
            return

        output_dir = os.path.dirname(output_image)
        if output_dir and not os.path.exists(output_dir):
            print(f"The directory '{output_dir}' does not exist. Please enter a valid directory.")
            return

        translator = Translator()  
        translator.black_and_white(input_image, output_image)
        print(f"Image successfully converted and saved to: {output_image}")

    except subprocess.CalledProcessError as e:
        print(f"Error converting image: {e}. Please try again.")
    except ValueError:
        print("Invalid input! Please check the entered data.")
    except FileNotFoundError:
        print("The input image path was not found. Please try again with a valid path.")

    print("\nProcess complete. Thank you for using the Black & White Converter!")

def run_length_encoding():
    print("\nWelcome to the Run-Length Encoding Tool!")
    print("----------------------------------------------------\n")

    print("Task 1: Run-Length Encoding")

    byte_sequence = input("Enter the byte sequence (e.g., 'wwwwaaadexxxxxx'): ").strip()

    if not byte_sequence:
        print("Error: Input sequence cannot be empty.")
        return

    try:
        translator = Translator()
        compressed = translator.run_length_encode(byte_sequence)
        print("Run-Length Encoded Output:", compressed)
    except ValueError:
        print("Invalid sequence! Please enter a valid byte sequence.")

    print("\nProcess complete. Thank you for using the Run-Length Encoding Tool!")

def dct_tool():
    print("\nWelcome to the DCT Encoding and Visualization Tool!")
    print("----------------------------------------------------\n")

    while True:
        print("\nTask 1: DCT Encoding and Visualization")

        choice = input("Enter 'sinewave', 'noisy', 'squarewave', 'impulse', 'step', or 'exit' to quit: ").strip().lower()

        if choice == 'exit':
            break  

        try:
            if choice == 'sinewave':
                t = np.linspace(0, 2 * np.pi, 1000)
                clean_signal = np.sin(t)
                noise = np.random.normal(0, 0.5, clean_signal.shape)
                input_data = clean_signal + noise  # Sinewave with noise
            elif choice == 'noisy':
                t = np.linspace(0, 2 * np.pi, 1000)
                input_data = np.sin(t) + np.random.normal(0, 0.5, t.shape)  # Noisy sinewave
            elif choice == 'squarewave':
                t = np.linspace(0, 2 * np.pi, 1000)
                input_data = np.sign(np.sin(t))  # Square wave
            elif choice == 'impulse':
                t = np.linspace(0, 2 * np.pi, 1000)
                input_data = np.zeros_like(t)
                input_data[500] = 1  # Single impulse at the middle
            elif choice == 'step':
                t = np.linspace(0, 2 * np.pi, 1000)
                input_data = np.ones_like(t)  # Step function
            else:
                print("Invalid input. Please enter 'sinewave', 'noisy', 'squarewave', 'impulse', 'step', or 'exit'.")
                continue  

            dct_processor = DCT(type_of_dct=2)
            encoded_data = dct_processor.encode(input_data)
            decoded_data = dct_processor.decode(encoded_data)
            dct_processor.visualize(input_data, encoded_data, decoded_data)

            continue_choice = input("Do you want to try another function? (y/n): ").strip().lower()
            if continue_choice == 'n':
                break
        except ValueError:
            print("Error: Invalid input or encoding problem.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("\nProcess complete. Thank you for using the DCT Tool!")



def dwt_tool():
    print("\nWelcome to the DWT Encoding and Visualization Tool!")
    print("----------------------------------------------------\n")

    while True:
        print("\nTask 1: DWT Encoding and Visualization")

        choice = input("Enter 'sinewave', 'noisy', 'squarewave', 'impulse', 'step', or 'exit' to quit: ").strip().lower()

        if choice == 'exit':
            break  

        try:
            if choice == 'sinewave':
                t = np.linspace(0, 2 * np.pi, 1000)
                clean_signal = np.sin(t)
                noise = np.random.normal(0, 0.5, clean_signal.shape)
                input_data = clean_signal + noise  # Sinewave with noise
            elif choice == 'noisy':
                t = np.linspace(0, 2 * np.pi, 1000)
                input_data = np.sin(t) + np.random.normal(0, 0.5, t.shape)  # Noisy sinewave
            elif choice == 'squarewave':
                t = np.linspace(0, 2 * np.pi, 1000)
                input_data = np.sign(np.sin(t))  # Square wave
            elif choice == 'impulse':
                t = np.linspace(0, 2 * np.pi, 1000)
                input_data = np.zeros_like(t)
                input_data[500] = 1  # Single impulse at the middle
            elif choice == 'step':
                t = np.linspace(0, 2 * np.pi, 1000)
                input_data = np.ones_like(t)  # Step function
            else:
                print("Invalid input. Please enter 'sinewave', 'noisy', 'squarewave', 'impulse', 'step', or 'exit'.")
                continue  

            dwt_encoder = DWT(wavelet_type='db1')
            encoded_data, approximation_coeffs, detail_coeffs = dwt_encoder.encode(input_data)
            decoded_data = dwt_encoder.decode(encoded_data)
            dwt_encoder.visualize(input_data, encoded_data, approximation_coeffs, detail_coeffs, decoded_data)

            continue_choice = input("Do you want to try another function? (y/n): ").strip().lower()
            if continue_choice == 'n':
                break
        except ValueError:
            print("Error: Invalid input or encoding problem.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("\nProcess complete. Thank you for using the DWT Tool!")


if __name__ == "__main__":
    main()