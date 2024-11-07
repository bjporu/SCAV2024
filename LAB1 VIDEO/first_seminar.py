import sys
import os
import numpy as np
import image
import subprocess
#Begin by defining the class  RGB_to_YUV. A python class is used for creating objects (instances) and can hold attributes and methods.

class Translator():

    # conversion_matrix = [[ 0.299,  0.587, 0.14],
    #                      [-0.299, -0.587, 0.886],
    #                      [0.701,  -0.587, -0.114]]

    def __init__(self, R,G,B): #The __init__ method is a constructor that is called each time an instance of the class is created.
        #self represents the instance of the class being created.
        self.R = R 
        self.G = G
        self.B = B
        
    def rgb_to_yuv(self): # https://www.cs.sfu.ca/mmbook/programming_assignments/additional_notes/rgb_yuv_note/RGB-YUV.pdf
        conversion_matrix = np.array([[ 0.299,  0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615,  -0.51499, -0.10001]])
        
        rgb = np.array([self.R, self.G, self.B])
        return np.matmul(conversion_matrix, rgb)

    def yuv_to_rgb(self, Y, U, V): # https://www.cs.sfu.ca/mmbook/programming_assignments/additional_notes/rgb_yuv_note/RGB-YUV.pdf
        conversion_matrix = np.array([[ 1,  0, 1.13983],
                         [1, -0.39465, -0.58060],
                         [1,  2.03211, 0]])
        
        yuv = np.array([Y, U, V])
        return np.matmul(conversion_matrix, yuv)
    
    def resize(self, width, height, input_image, output_image): #https://trac.ffmpeg.org/wiki/Scaling
        command = [
            'ffmpeg',
            '-i', input_image,
            '-vf', f'scale={width}:{height}',
             '-frames:v', '1',
            output_image
            ]
        
        subprocess.run(command, check=True)

def main():
    print("\nWelcome to the RGB-YUV Converter and Image Resizer!")
    print("----------------------------------------------------\n")

    # Task 1: Ask the user for RGB values
    print("Task 1: RGB to YUV Conversion")
    try:
        R = int(input("Enter the R (Red) value (0-255): "))
        G = int(input("Enter the G (Green) value (0-255): "))
        B = int(input("Enter the B (Blue) value (0-255): "))
    except ValueError:
        print("Invalid input! Please enter integers between 0 and 255.")
        return
    
    translator = Translator(R, G, B)
    YUV = translator.rgb_to_yuv()
    print(f"\nConverted RGB({R}, {G}, {B}) to YUV: {YUV}")
    
    # Task 2: Ask the user if they want to convert back to RGB
    print("\nTask 2: YUV to RGB Conversion")
    convert_back = input("Do you want to convert back to RGB (y/n)? ").strip().lower()
    
    if convert_back == 'y':
        Y = float(input("Enter the Y value: ")) 
        U = float(input("Enter the U value: ")) 
        V = float(input("Enter the V value: "))
        
        RGB = translator.yuv_to_rgb(Y, U, V)
        print(f"Converted YUV({Y}, {U}, {V}) back to RGB: {RGB}")
    
    # Task 3: Ask the user for image resizing
    print("\nTask 3: Image Resizing")
    resize_image = input("Do you want to resize an image (y/n)? ").strip().lower()
    
    if resize_image == 'y':
        while True:  # Loop to retry if there is an error
            try:
                input_image = input("Enter the path of the input image: ").strip()
                
                # Check if input image exists
                if not os.path.exists(input_image):
                    print(f"The path '{input_image}' does not exist. Please enter a valid path.")
                    continue  # Retry the input
                
                output_image = input("Enter the full path where you want to save the resized image (including filename and format, e.g., 'output/image_resized.jpeg'): ").strip()
                
                # Ensure the user has provided a filename with a valid extension
                if not output_image.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.gif')):
                    print("Please provide a valid image format in the output filename (e.g., .jpeg, .png).")
                    continue  # Retry the input
                
                # Check if output directory exists
                output_dir = os.path.dirname(output_image)
                if output_dir and not os.path.exists(output_dir):
                    print(f"The directory '{output_dir}' does not exist. Please enter a valid directory.")
                    continue  # Retry the input
                
                width = int(input("Enter the desired width for the image: "))
                height = int(input("Enter the desired height for the image: "))
                
                # Attempt resizing the image
                translator.resize(width, height, input_image, output_image)
                print(f"Image resized successfully! Output saved to: {output_image}")
                break  # Exit the loop if successful
                
            except subprocess.CalledProcessError as e:
                print(f"Error resizing image: {e}. Please try again.")
            except ValueError:
                print("Invalid input! Please enter integers for width and height.")
            except FileNotFoundError:
                print("The input image path was not found. Please try again with a valid path.")
    
    print("\nProcess complete. Thank you for using the RGB-YUV Converter and Image Resizer!")

if __name__ == "__main__":
    main()