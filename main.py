from calendar import c
import matplotlib.pyplot as plt
import matplotlib.colors as cr_liner
import numpy as np
from scipy.fftpack import dct as dct 
from scipy.fftpack import idct as idcts
from decodec_jpeg import decodec,upsampling
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


def codec(img, channel=None, grey=None,activate_plot=0):
    channels = {}
    rgb = ['Red', 'Green', 'Blue']
    if activate_plot == 0:
        for color in range(3):
            get_channels(channels, img, color)
            if channel is None:
                plot_image_colormap(channels[rgb[color]], colormap(rgb[color]))
                plot_image_colormap(channels[rgb[color]], colormap(rgb[color], 1))
        if channel is not None:
            plot_image_colormap(channels[channel], colormap(rgb[channel], grey))
        aux=padding(img)
        plot_image_colormap(aux)
        a=ycbr(aux, 2)
    elif activate_plot == 1:
        aux=padding(img)
        plot_image_colormap(aux)
        a=ycbr(img)
    elif activate_plot == 2:
        aux=padding(img)
        a=ycbr(aux, activate_plot)
    else:
        aux=padding(img)
        a=ycbr(aux)

    return a



###############################################################################################
def padding(img,pad=16):
    try:
        p1 = img[:, :, 0]
        p2 = img[:, :, 1]
        p3 = img[:, :, 2]
        print('------------------------------')
        print("Original dim = ", img.shape)
        print('------------------------------')
        r, c = p1.shape
        r_new, c_new = p1.shape
        if r%pad!=0:
            repeat_arr=np.ones(r,dtype="uint8")
            repeat_arr[r-1]=pad-(r % pad)+1
            p1 = np.repeat(p1, repeat_arr,axis=0)
            p2 = np.repeat(p2, repeat_arr,axis=0)
            p3 = np.repeat(p3, repeat_arr,axis=0)
            r_new,c_new=p1.shape
        if c%pad!=0:
            repeat_arr=np.ones(c,dtype="uint8")
            repeat_arr[c-1]=pad-c % pad+1
            p1 = np.repeat(p1, repeat_arr,axis=1)
            p2 = np.repeat(p2, repeat_arr,axis=1)
            p3 = np.repeat(p3, repeat_arr,axis=1)
            r_new,c_new=p1.shape
        padded_img = np.zeros((r_new,c_new, 3))
        padded_img[:, :, 0] = p1
        padded_img[:, :, 1] = p2
        padded_img[:, :, 2] = p3
        return padded_img
    except:
        p1 = img[:, :]
        r, c = p1.shape
        r_new, c_new = p1.shape
        p1 = img[:, :] 
        if r%pad!=0:
            repeat_arr=np.ones(r,dtype="uint8")
            repeat_arr[r-1]=pad-r % pad+1
            p1 = np.repeat(p1, repeat_arr,axis=0)
            r_new,c_new=p1.shape
        if c%pad!=0:
            repeat_arr=np.ones(c,dtype="uint8")
            repeat_arr[c-1]=pad-c % pad+1
            p1 = np.repeat(p1, repeat_arr,axis=1)
            r_new,c_new=p1.shape
        padded_img = np.zeros((r_new,c_new))
        padded_img[:, :] = p1
        
        return padded_img


def padding_decoder(original_l, original_c, padded_img):
    unpadded_img = padded_img[: original_l, : original_c, :]
    print("Unpadded dim = ", unpadded_img.shape)
    plt.figure()
    plt.imshow(unpadded_img)
    plt.axis('off')
    plt.show()

    return unpadded_img


###############################################################################################
def ycbr(img, activate_plot=0):
    """
    plt.figure()
    img = rgb_to_ycbr(img)
    plt.imshow(img)
    plt.axis('off')

    plt.figure()
    img = ycbr_to_rgb(img)
    plt.imshow(img)
    """
    img = rgb_to_ycbr(img)
    if activate_plot == 2:
        for i in range(3):
            plot_image_colormap(img[:, :, i], ycbr_colormap(i))
    downsampling(img[:, :, 0],img[:, :, 1],img[:, :, 2])
    return img


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


def rgb_to_ycbr(img):
    formula = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    ycbr_img = img.dot(formula.T)  
    ycbr_img[:, :, [1, 2]] += 128
    #ycbr_to_rgb(ycbr_img)
    return ycbr_img



###############################################################################################
def func_dct(x,type,title):
    X_dct = dct(dct(x, norm="ortho").T, norm="ortho").T # DCT
    plt.figure()
    plt.title("Dct " +type+" "+title)
    plt.imshow(np.log(np.abs(X_dct) + 0.0001),colormap(grey=1))
    return X_dct


def downsampling(y, cb, cr,type="4:2:2"):
    arr = [y, cb, cr]
    arr_tit = ["y", "cb", "cr"]
    for i in range (3):
        plt.figure()
        arr[i] = get_dchannels(arr[i], type)
        plt.title("Downsampling " +type+" "+arr_tit[i])
       
        plt.imshow(arr[i],colormap(grey=1))
        func_dct(arr[i],type,arr_tit[i])
    dct_bsxbs(y,8)
    dct_bsxbs(cb,8)
    dct_bsxbs(cr,8)
    return arr



def dct_bsxbs(test_image,bs):
    if(bs==64):
        test_image=padding(test_image,64)
        print(test_image.shape)
    
    l,c=test_image.shape
    aux_l = int(l/bs)
    aux_c = int(c/bs)
    sliced = test_image.reshape(aux_l*aux_c,bs,bs)
    for i in range(len(sliced)):
        sliced[i] = dct(dct(sliced[i], norm="ortho").T, norm="ortho").T
    aux = sliced.reshape(l,c)
    plt.figure()
    plt.title("dct blocos de "+str(bs)+"x"+str(bs))
    plt.imshow(np.log(np.abs(aux) + 0.0001),colormap(grey=1))
    return aux
    

           


#X_dct = dct(dct(X, norm=”ortho”).T, norm=”ortho”).T
def get_dchannels(dchanel, type):
    if type == "4:2:0":
        return ds_4_2_0(dchanel)
    if type == "4:2:2":
        return ds_4_2_2(dchanel)
    else:
        return None



def ds_4_2_2(img):  # 4:2:2
    C = img.copy()
    C = C[:, ::2] #cv2.resize??
    return C


def ds_4_2_0(img):  # 4:2:0
    # Todos os 2 elementos vao ser iguais aos elementos do lado/acima logo são removidos .
    B = img.copy()  
    # Vertical
    B = B[::2, :] #cv2.resize??
    # Horizontal
    B = B[:, ::2]
    return B



###############################################################################################
def main():
    img = plt.imread('imagens/barn_mountains.bmp')
    original_c,original_l,c=img.shape
    img_v = codec(img,activate_plot=2)
    decodec(original_l,original_c,img_v)
    # padding_decoder(img, padding(img))
    # plot_image_colormap(channels)
    # ycbr(img)
    plt.show()


if __name__ == "__main__":
    main()