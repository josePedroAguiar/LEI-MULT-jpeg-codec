import matplotlib.pyplot as plt
import matplotlib.colors as cr_liner
import numpy as np
from scipy.fftpack import dct as dct 
from scipy.fftpack import idct as idct

import math as m
bs=16
original_l=0
original_c=0
def original(img):  # Plot Original
    # plt.figure()
    plt.axis('off')
    plt.imshow(img)
    print(img.shape)  # (linhas, colunas, canais) canais = 3 (r,g,b)


def colormap(color='', grey=None):
    if grey is not None:
        cm = cr_liner.LinearSegmentedColormap.from_list("Grey", [(0, 0, 0), (1, 1, 1)], N=255)
    else:
        if color == 'Red':
            cm = cr_liner.LinearSegmentedColormap.from_list("Red", [(0, 0, 0), (1, 0, 0)], N=255)
        elif color == 'Green':
            cm = cr_liner.LinearSegmentedColormap.from_list("Green", [(0, 0, 0), (0, 1, 0)], N=255)
        elif color == 'Blue':
            cm = cr_liner.LinearSegmentedColormap.from_list("Blue", [(0, 0, 0), (0, 0, 1)], N=255)
        # plot_image_colormap(channels,cm)
        else:
            return
    return cm

def iycbr(img):
    """
    plt.figure()
    img = rgb_to_ycbr(img)
    plt.imshow(img)
    plt.axis('off')

    plt.figure()
    img = ycbr_to_rgb(img)
    plt.imshow(img)
    """
    aux=upsampling(img[:, :, 0],img[:, :, 1],img[:, :, 2])
    img = ycbr_to_rgb(aux)
    return img

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



def decodec(original_l, original_c, img_jpg):
    img=padding_decoder(original_l, original_c,img_jpg)
    plot_image_colormap(img)
    a=iycbr(img)
    return a


###############################################################################################
def padding_decoder(original_l, original_c, padded_img):
    unpadded_img = padded_img[: original_l, : original_c, :]
    print("Unpadded dim = ", unpadded_img.shape)
    plt.figure()
    plt.imshow(unpadded_img)
    plt.axis('off')
    plt.show()

    return unpadded_img


###############################################################################################
def ycbr_colormap(channel):
    if channel == 0:
        cm = cr_liner.LinearSegmentedColormap.from_list("y", [(0, 0, 0), (1, 1, 1)], N=255)
    elif channel == 1:
        cm = cr_liner.LinearSegmentedColormap.from_list("cb", [(0, 0.5, 0), (0, 0, 0.5)], N=255)
    elif channel == 2:
        cm = cr_liner.LinearSegmentedColormap.from_list("cr", [(0, 0.5, 0), (0.5, 0, 0)], N=255)
    else:
        return
    return cm


def ycbr_to_rgb(img):
    formula = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    formula = np.linalg.inv(formula)
    rgb = np.array(img)
    print(rgb.shape)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.round()
    rgb = rgb.dot(formula.T)  
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


###############################################################################################
def func_idct(x,type,title):
    X_idct = idct(idct(x, norm="ortho").T, norm="ortho").T # DCT
    plt.figure()
    plt.title("Dct " +type+" "+title)
    plt.imshow(X_idct,colormap(grey=1))
    return X_idct

def upsampling(y, cb, cr,type="4:2:2"):
    arr = [y, cb, cr]
    print("_____________")
    print(arr)
    print("_____________")
    arr_tit = ["y", "cb", "cr"]
    for i in range (3):
        plt.figure()
        arr[i] = get_upchannels(arr[i],type)
        x=func_idct(arr[i],type,arr_tit[i])

        plt.title("Upsampling " +type+" "+arr_tit[i])
        plt.imshow(arr[i],colormap(grey=1))
        #idct_bsxbs(arr[i],8)

  
    return np.dstack((y,cb,cr))
def idct_bsxbs(test_image,bs):   
    if(bs == 64):
        test_image=padding_decoder(test_image,64)
        print(test_image.shape)
    l,c=test_image.shape
    aux_l = int(l/bs)
    aux_c = int(c/bs)
    sliced = test_image.reshape(aux_l*aux_c,bs,bs)
    for i in range(len(sliced)):
        sliced[i] = idct(idct(sliced[i], norm="ortho").T, norm="ortho").T
    aux = sliced.reshape(l,c)
    plt.figure()
    plt.title("idct blocos de "+str(bs)+"x"+str(bs))
    plt.imshow(np.log(np.abs(aux) + 0.0001),colormap(grey=1))
    return aux
           

def get_upchannels(dchanel, type):
    if type == "4:2:0":
        return reverse_ds_4_2_0(dchanel)
    if type == "4:2:2":
        print("$$$$$$$$$$$$$")
        print(dchanel)
        return reverse_ds_4_2_2(dchanel)
    else:
        return None


def reverse_ds_4_2_2(img):
    print (img)
    return np.repeat(img, 2, axis=1)


def reverse_ds_4_2_0(img):
    return np.repeat(np.repeat(img, 2, axis=0), 2, axis=0)


###############################################################################################
