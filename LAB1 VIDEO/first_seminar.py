import sys
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
        
        yuv = np.array([self.R, self.G, self.B])
        return np.matmul(conversion_matrix, yuv)
    
    def resize(self, width, height, quality, input_image, output_image): #https://stackoverflow.com/questions/28806816/use-ffmpeg-to-resize-image
        command = [
            'ffmpeg',
            '-f', 'mjpeg',
            '-i', input_image,
            '-vf', f'scale={width}:{height}',
            '-q:v', str(quality),  # Set the quality for JPEG output
            output_image
            ]
        
        subprocess.run(command, check=True)

def main():
    #cridar coses
    translator = Translator(100, 150, 200)
    print("YUV:", translator.rgb_to_yuv())

    input_image = r'C:\Users\pabli\OneDrive\Escritorio\Uni 2025\SCAV\snoop_dogg.jpeg'
    output_image = r'C:\Users\pabli\OneDrive\Escritorio\Uni 2025\SCAV\output.jpeg' 
    translator.resize(300, 200, 30, input_image, output_image)

    
if __name__ == "__main__":
    main()




