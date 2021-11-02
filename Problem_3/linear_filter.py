#!/usr/bin/env python3

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########
    I_row, I_col, I_dep = np.shape(I)
    G = np.zeros([I_row,I_col])
    row, col, dep = np.shape(F)
    #print(row)
    #print(col)
    #print(dep)
    row_pad = int(np.floor(row/2))
    col_pad = int(np.floor(col/2))
    new_I = np.pad(I,((row_pad,row_pad),(col_pad,col_pad),(0,0)), mode="constant")
    #print(new_I)

    #f = np.reshape(F,(1,row*col*dep))
    f = F.flatten().T
    for i in range(I_row):
        for j in range(I_col):
            #t = np.reshape(new_I[:,j:j+row,i:i+col])
            t = new_I[i:i+row,j:j+col,0:dep].flatten()
            G[i,j] = np.sum(f*t)

    return G

    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########
    I_row, I_col, I_dep = np.shape(I)
    G = np.zeros([I_row,I_col])
    row, col, dep = np.shape(F)
    #print(row)
    #print(col)
    #print(dep)
    row_pad = int(np.floor(row/2))
    col_pad = int(np.floor(col/2))
    new_I = np.pad(I,((row_pad,row_pad),(col_pad,col_pad),(0,0)), mode="constant")
    #print(new_I)

    #f = np.reshape(F,(1,row*col*dep))
    f = F.flatten().T
    for i in range(I_row):
        for j in range(I_col):
            #t = np.reshape(new_I[:,j:j+row,i:i+col])
            t = new_I[i:i+row,j:j+col,0:dep].flatten()
            G[i,j] = np.sum(f*t)/(np.linalg.norm(f)*np.linalg.norm(t))

    return G
    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 3, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        #corr_img = norm_cross_corr(filt, test_card)
        stop = time.time()
        print('Correlation function runtime:', stop - start, 's')
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)
        #show_save_corr_img("norm_corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
