import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


def original(img):  # Plot Original
    # plt.figure()
    plt.axis('off')
    plt.imshow(img)
    print(img.shape)  # (linhas, colunas, canais) canais = 3 (r,g,b)


def colormap(color='', grey=None):
    if grey is not None:
        cm = clr.LinearSegmentedColormap.from_list("Grey", [(0, 0, 0), (1, 1, 1)], N=255)
    else:
        if color == 'Red':
            cm = clr.LinearSegmentedColormap.from_list("Red", [(0, 0, 0), (1, 0, 0)], N=255)
        elif color == 'Green':
            cm = clr.LinearSegmentedColormap.from_list("Green", [(0, 0, 0), (0, 1, 0)], N=255)
        elif color == 'Blue':
            cm = clr.LinearSegmentedColormap.from_list("Blue", [(0, 0, 0), (0, 0, 1)], N=255)
        # plot_image_colormap(channels,cm)
        else:
            return
    return cm


def plot_image_colormap(img, cm=None):
    plt.figure()
    plt.axis('off')
    if cm is not None:
        plt.imshow(img, cm)
    else:
        plt.imshow(img)


def get_channels(channels, img, channel):
    rgb = ['Red', 'Green', 'Blue']
    channels[rgb[channel]] = img[:, :, channel]


def codec(img, channel=None, grey=None):
    channels = {}
    rgb = ['Red', 'Green', 'Blue']
    for color in range(3):
        get_channels(channels, img, color)
        if channel is None:
            plot_image_colormap(channels[rgb[color]], colormap(rgb[color]))
            plot_image_colormap(channels[rgb[color]], colormap(rgb[color], 1))
    if channel is not None:
        plot_image_colormap(channels[channel], colormap(rgb[channel], grey))
    return channels


def decodec(channels):
    plt.figure()
    plt.title("Decoder")
    plt.axis('off')
    plt.imshow(np.dstack((channels['Red'],
                          channels['Green'],
                          channels['Blue'])))
    return


###############################################################################################
def padding(img):
    p1 = img[:, :, 0]
    p2 = img[:, :, 1]
    p3 = img[:, :, 2]
    print('------------------------------')
    print("Original dim = ", img.shape)
    print('------------------------------')
    r, c = p1.shape
    while (r % 16) != 0:
        p1 = np.vstack([p1, p1[-1, :]])
        p2 = np.vstack([p2, p2[-1, :]])
        p3 = np.vstack([p3, p3[-1, :]])
        r, c = p1.shape
    while (c % 16) != 0:

        p1 = np.column_stack([p1, p1[:, -1]])
        p2 = np.column_stack([p2, p2[:, -1]])
        p3 = np.column_stack([p3, p3[:, -1]])
        r, c = p1.shape

    padded_img = np.zeros((r, c, 3), dtype=np.uint8)
    padded_img[:, :, 0] = p1.astype(np.uint8)
    padded_img[:, :, 1] = p2.astype(np.uint8)
    padded_img[:, :, 2] = p3.astype(np.uint8)
    print("Padded dim = ", padded_img.shape)
    plt.figure()
    plt.imshow(np.dstack((p1, p2, p3)))
    plt.axis('off')
    plt.show()
    return padded_img


def padding_decoder(img, padded_img):
    nl, nc, color = img.shape

    unpadded_img = padded_img[:nl, :nc, :]
    print("Unpadded dim = ", unpadded_img.shape)
    plt.figure()
    plt.imshow(unpadded_img)
    plt.axis('off')
    plt.show()

    return


###############################################################################################


def ycbcr(img):
    """
    plt.figure()
    img = rgb_to_ycbcr(img)
    plt.imshow(img)
    plt.axis('off')

    plt.figure()
    img = ycbcr_to_rgb(img)
    plt.imshow(img)"""
    img = rgb_to_ycbcr(img)
    for i in range(3):
        # ycbcr_colormap(i)
        plot_image_colormap(img[:, :, i], ycbcr_colormap(i))


def ycbcr_colormap(channel):
    if channel == 0:
        cm = clr.LinearSegmentedColormap.from_list("y", [(0, 0, 0), (1, 1, 1)], N=255)
    elif channel == 1:
        cm = clr.LinearSegmentedColormap.from_list("cb", [(0, 0.5, 0), (0, 0, 0.5)], N=255)
    elif channel == 2:
        cm = clr.LinearSegmentedColormap.from_list("cr", [(0, 0.5, 0), (0.5, 0, 0)], N=255)
    else:
        return
    return cm


def rgb_to_ycbcr(img):
    formula = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    ycbcr_img = img.dot(formula.T)  # duvida porque que temos de usar a transposta
    ycbcr_img[:, :, [1, 2]] += 128
    return ycbcr_img


def ycbcr_to_rgb(img):
    formula = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    formula = np.linalg.inv(formula)
    rgb = img.astype(float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(formula.T)  # duvida porque que temos de usar a transposta
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


###############################################################################################
def main():
    plt.close('all')
    img = plt.imread('imagens/logo.bmp')
    # channels = codec(img)
    # decodec(channels)
    padding_decoder(img, padding(img))
    # plot_image_colormap(channels['Red'],map('Red'))
    # ycbcr(img)
    plt.show()


if __name__ == "__main__":
    main()
