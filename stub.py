from PIL import Image, ImageQt
import numpy as np

def get_image():
    linspace = np.linspace(0,255,1024)
    linspace = linspace.round()
    x = np.tile(linspace, (720,1))
    # print(x[0])
    x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
    # x = x[:, :, np.newaxis]
    # x = np.concatenate((x,x,x), axis=2)

    img = Image.fromarray(x, mode="RGB")
    img = ImageQt.ImageQt(img)
    # img.show()
    return img   


get_image()