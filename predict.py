import numpy as np
import os

from PIL import Image

from siamese import Siamese
#OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#解决上面这个问题所需要用到的代码（os）
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__":
    model = Siamese()
        
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        probability = model.detect_image(image_1,image_2)
        print(probability)
