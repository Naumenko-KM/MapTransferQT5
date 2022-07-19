import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from generator import Generator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_image(path: str, mode: int) -> np.array:
    """
    Open image, feed it to generator.
    - path - path to image
    - mode - [0, 1, 2] -> [ifrared, win-sum, sum-win]
    Return generated image in numpy array [height, width, 3]
    """
    img_0 = Image.open(path)
    img_0_cropped, img_size = crop_image(img_0)
    if mode == 0:
        img_1_cropped = generate_image_vis2inf(img_0_cropped)
    elif mode == 1:
        img_1_cropped = generate_image_sum2win(img_0_cropped)
    elif mode == 2:
        img_1_cropped = generate_image_win2sum(img_0_cropped)
    img_1 = uncrop_image(img_1_cropped, img_size)
    return img_1


def crop_image(img: Image.Image) -> tuple((np.array, tuple)):
    """
    Return cropped image and image size
    [height, width, 3] -> [B, 256, 256, 3], (height//256, width//256)
    """
    batch = []
    for x in range(0, img.size[0]-256, 256):
        for y in range(0, img.size[1]-256, 256):
            batch.append(np.array(img.crop((x, y, x+256, y+256))))
            # [256, 256, 3]
    batch = np.array(batch)  # [B, 256, 256, 3]
    return batch, (img.size[0]//256, img.size[1]//256)
    # return np.ones((4, 256, 256, 3))


def uncrop_image(batch: np.array, img_size: tuple) -> np.array:
    """
    Return image from batch of crops
    [B, 256, 256, 3], (height//256, width//256) -> [height, width, 3] 
    """
    batch = batch.reshape((img_size[0], img_size[1], batch.shape[1],
                           batch.shape[2], batch.shape[3]))
    stacked = stack_batch_v(batch[0], img_size[1])
    for i in range(1, img_size[0]):
        stacked = np.hstack((stacked,
                             stack_batch_v(batch[i], img_size[1])))
    return stacked


def stack_batch_v(h_batch, size):
    v_stacked = h_batch[0]
    for i in range(1, size):
        v_stacked = np.vstack((v_stacked, h_batch[i]))
    return v_stacked


def generate_image_vis2inf(batch, batch_size=64) -> np.array: 
    """
    Return generated cropped image
    """
    model = Generator()
    model.to(DEVICE)
    checkpoint = torch.load('gen.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    batch = batch / 255
    batch = torch.Tensor(batch).to(DEVICE)
    batch = batch.permute(0, 3, 1, 2)
    with torch.no_grad():
        y_fake = model(batch[:batch_size])
        for i in range(1, batch.shape[0]//batch_size + 1):
            y_fake_tmp = model(batch[i*batch_size:(i+1)*batch_size])
            y_fake = torch.cat((y_fake, y_fake_tmp), 0)
            print(y_fake.shape)

    y_fake = y_fake.permute(0, 2, 3, 1)
    y_fake = y_fake.detach().cpu().numpy()
    # print(y_fake)
    y_fake = (y_fake * 255).astype(np.uint8)
    y_fake[:, :, :, 1:] = 0
    return y_fake


def generate_image_sum2win(batch, batch_size=64) -> np.array: 
    pass


def generate_image_win2sum(batch, batch_size=64) -> np.array: 
    pass


if __name__ == "__main__":
    path = '20.0_49.0_vis.jpeg'
    y = generate_image(path)
    # y = Image.open(path)
    # y = np.array(y)
    print(y.shape)
    plt.imshow(y)
    plt.show()
