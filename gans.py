import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from generator import Generator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_image(path: str) -> np.array:
    """
    Open image, feed it to generator.
    Return generated image in numpy array [height, width, 3]
    """
    img_0 = Image.open(path)
    img_0_batch, img_size = img_to_batch(img_0)
    img_1_batch = generate_batch(img_0_batch)
    img_1 = img_from_batch(img_1_batch, img_size)
    return img_1


def img_to_batch(img: Image.Image) -> tuple((np.array, tuple)):
    """
    Return batch of image's crops and image size
    [height, width, 3] -> [B, 256, 256, 3]
    """
    batch = []
    for x in range(0, img.size[0]-256, 256):
        for y in range(0, img.size[1]-256, 256):
            batch.append(np.array(img.crop((x, y, x+256, y+256))))
            # [256, 256, 3]
    batch = np.array(batch)  # [B, 256, 256, 3]
    return batch, (img.size[0]//256, img.size[1]//256)
    # return np.ones((4, 256, 256, 3))


def img_from_batch(batch: np.array, img_size: tuple) -> np.array:
    """
    Return image from batch of crops
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


def generate_batch(batch):
    """
    Return generated batch
    """
    model = Generator()
    model.to(DEVICE)
    checkpoint = torch.load('gen.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    batch = torch.Tensor(batch).to(DEVICE)
    batch = batch.permute(0, 3, 1, 2)
    y_fake = model(batch)
    y_fake = y_fake.permute(0, 2, 3, 1)
    y_fake = y_fake.detach().cpu().numpy()
    return y_fake


if __name__ == "__main__":
    path = '20.0_49.0_vis.jpeg'
    Image.open(path)
    y = generate_image(path)
    plt.imshow(y)
