import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from CONFIG.config import *

def plot_images(images, titles=[], gridx=None, gridy=None, figsize=(10, 10)): 
    n = len(images)
    gridx = np.sqrt(n) if gridx is None else gridx
    gridy = np.sqrt(n) if gridy is None else gridy
    titles = [''] * n if len(titles)<n else titles

    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(gridx, gridy, i+1)
        cmap = 'gray' if images[i].ndim==2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.xticks([]);plt.yticks([])
        plt.title(titles[i])
    
    plt.show()

def transform_points(pts: np.ndarray, M: np.ndarray):
    '''
    pts shape: (N, 2)
    '''
    out = M.dot(np.vstack((pts.T, np.ones(pts.shape[0]))))
    out = out[:2, :] / out[2, :]
    return out


# Color Space Selection

def Saturation_thresh(img, thresh=(0, 255)):
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    binary_im = np.ones_like(s)
    binary_im[(s > thresh[0]) & (s < thresh[1])] = 0
    return binary_im


def L_lab_thresh(img, thresh=(0, 255)):
    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    b = lab[:, :, 0]
    binary_im = np.ones_like(b)
    binary_im[(b >= thresh[0]) & (b <= thresh[1])] = 0
    return binary_im


def B_lab_thresh(img, thresh=(0, 255)):
    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    b = lab[:, :, 2]
    binary_im = np.ones_like(b)
    binary_im[(b >= thresh[0]) & (b <= thresh[1])] = 0
    return binary_im


def luv_thresh(img, thresh=(0, 255)):
    luv = cv.cvtColor(img, cv.COLOR_RGB2LUV)
    v = luv[:, :, 2]
    binary_im = np.ones_like(v)
    binary_im[(v > thresh[0]) & (v < thresh[1])] = 0
    return binary_im


def combined(*images):
    c = np.zeros_like((images[0]))
    for image in images:
        c[image == 1] = 1
    return c

# Histogram of the bottom half of the image to find the lane lines.


def histogram(img):
    return np.sum(img[img.shape[0]//2:, :], axis=0)


# Find the lane pixels using windowing
def find_lane_pixel_windowing(  binary_warped, 
                                fit_original_img, 
                                nwindows=9,
                                margin=100,
                                minpix=50, 
                                Minv: np.ndarray=Minv):
    '''
    Find the lane pixels using windowing
    Args: 
        binary_warped: Binary image after applying perspective transform..
        nwindows: number of windows to split the image into.
        margin: width of the windows +/- margin.
        minpix: minimum number of pixels found to recenter the window.
        fit_original_img: If True, the fit will be done in the original image.default: False.

    Returns:
        left_fit: left lane polynomial coefficients at the form [a,b,c] where y = ax^2 + bx + c.
        right_fit: right lane polynomial coefficients at the form [a,b,c] where y = ax^2 + bx + c.
    '''
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    h = histogram(binary_warped)

    # Starting x position for left and right lane
    midpoint = np.int(h.shape[0]//2)
    leftx_current = np.argmax(h[:midpoint])
    rightx_current = np.argmax(h[midpoint:]) + midpoint

    # We are going to split the image into nwindows,
    # so each will have a height of total height / nwindows
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify all x and y positions of non-zero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create empty lists to receive left and right lane pixel indices in each window
    left_lane_inds = []
    right_lane_inds = []

    # Window sliding
    for window in range(nwindows):
        # Identify the right and left windows
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv.rectangle(out_img, (win_xleft_low, win_y_low),
                     (win_xleft_high, win_y_high),   (0, 255, 0), 2)
        cv.rectangle(out_img, (win_xright_low, win_y_low),
                     (win_xright_high, win_y_high), (255, 0, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            # Calculate the next x center of the left window
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            # Calculate the next x center of the right window
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices of lines pixels
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if not fit_original_img:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        # Fit new polynomials to x,y in world space
        leftx_ws, lefty_ws = transform_points(
            np.vstack([leftx, lefty]).T, Minv)
        left_fit_ws = np.polyfit(lefty_ws, leftx_ws, 2)

        rightx_ws, righty_ws = transform_points(
            np.vstack([rightx, righty]).T, Minv)
        right_fit_ws = np.polyfit(righty_ws, rightx_ws, 2)

        return left_fit_ws, right_fit_ws

    return left_fit, right_fit, out_img


# We are going to use the previous frame to find the lane lines
# Instead of using the windowing method, we will use a new window,
# the boundaries of this new window is going to be the previous fit - margin , fit + margin
def fit_using_previous_fit(binary_warped, left_fit, right_fit,
                           fit_original_img=False,
                           margin=100):
    '''
    Fit the lane lines using the previous fit
    Args:
        binary_warped: Binary image after applying perspective transform.
        left_fit: Previous left fit polynomial at the form [a,b,c] where y = ax^2 + bx + c.
        right_fit: Previous right fit polynomial at the form [a,b,c] where y = ax^2 + bx + c.
        fit_original_img: If True, the fit will be done in the original image. Default is False
        margin: Margin of window from the previous fit. Default is 100.
    
    Returns:
        left_fit: New left fit polynomial at the form [a,b,c] where y = ax^2 + bx + c.
        right_fit: New right fit polynomial at the form [a,b,c] where y = ax^2 + bx + c.
        out_img: Blank Warped image with the windows drawn.
    '''

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Identify Region of interest
    leftx_low = (left_fit[0]*(nonzeroy**2) + left_fit[1]
                 * nonzeroy + left_fit[2] - margin)
    leftx_high = (left_fit[0]*(nonzeroy**2) + left_fit[1]
                  * nonzeroy + left_fit[2] + margin)

    rightx_low = (right_fit[0]*(nonzeroy**2) +
                  right_fit[1]*nonzeroy + right_fit[2] - margin)
    rightx_high = (right_fit[0]*(nonzeroy**2) +
                   right_fit[1]*nonzeroy + right_fit[2] + margin)

    # within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > leftx_low) & (nonzerox < leftx_high))
    right_lane_inds = ((nonzerox > rightx_low) & (nonzerox < rightx_high))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if not fit_original_img:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        # Fit new polynomials to x,y in world space
        leftx_ws, lefty_ws = transform_points(
            np.vstack([leftx, lefty]).T, Minv)
        left_fit_ws = np.polyfit(lefty_ws, leftx_ws, 2)

        rightx_ws, righty_ws = transform_points(
            np.vstack([rightx, righty]).T, Minv)
        right_fit_ws = np.polyfit(righty_ws, rightx_ws, 2)

        return left_fit_ws, right_fit_ws

    # Visualization
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Identify window boundaries in x and y (and right and left)
    left_line_window1 = np.array(
        [np.transpose(np.vstack([leftx_low, nonzeroy]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([leftx_high, nonzeroy])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array(
        [np.transpose(np.vstack([rightx_low, nonzeroy]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([rightx_high, nonzeroy])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, left_fit, right_fit

def draw_lines(img:np.ndarray, left_fit: list, right_fit:list):
    '''
    Draw lines on the image: 
    Args:
        img: image to draw on
        left_fit: left line fit at the form of [a,b,c] where y = ax^2 + bx + c
        right_fit: right line fit at the form of [a,b,c] where y = ax^2 + bx + c

    Returns:
        img: image with the lines drawn on it
    '''
    y = np.linspace((img.shape[0]-1)//1.6, (img.shape[0]-1), img.shape[0])
    x_left_fit_o = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    x_right_fit_o = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]

    left_line_pts = np.array([np.transpose(np.vstack([x_left_fit_o, y]))])
    right_line_pts = np.array(
        [np.flipud(np.transpose(np.vstack([x_right_fit_o, y])))])
    all_pts = np.hstack((left_line_pts, right_line_pts))

    blank = np.zeros_like(img)
    cv.fillPoly(blank, np.int_([all_pts]), (0, 255, 0))
    out_img = cv.addWeighted(img, 1, blank, 0.3, 0)
    cv.polylines(out_img, np.int_([left_line_pts]), False, (0, 255, 0), 5)
    cv.polylines(out_img, np.int_([right_line_pts]), False, (0, 255, 0), 5)
    
    return out_img

def pipeline(img,
             first_frame=False, 
             left_fit=[], 
             right_fit=[], 
             M:np.ndarray=M, 
             Minv:np.ndarray=Minv):
    '''
    Pipeline to process the image and find the lane lines

    Args:
        img: image to process
        first_frame: if True, the image is processed from scratch
        left_fit: left line fit at the form of [a,b,c] where y = ax^2 + bx + c
        right_fit: right line fit at the form of [a,b,c] where y = ax^2 + bx + c

    Returns:
        img: image with the lines drawn on it
        left_fit: left line fit at the form of [a,b,c] where y = ax^2 + bx + c
        right_fit: right line fit at the form of [a,b,c] where y = ax^2 + bx + c

    Raises:
        Exception: if first_frame is False and left_fit and right_fit are None
    '''

    # if (not first_frame & (left_fit) & (right_fit is None)):
    #     raise Exception("first_frame must be True if left_fit and right_fit are None")
    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # undistort image
    img = cv.undistort(img, mtx, dist, None, mtx)

    # prespective transform
    warped_im = cv.warpPerspective(img, M, img.shape[1::-1], flags=cv.INTER_LINEAR)

    # color space thresholding
    # s = Saturation_thresh(warped_im, S_THRESH)
    l = L_lab_thresh(warped_im, L_LAB_THRESH)
    b = B_lab_thresh(warped_im, B_LAB_THRESH)
    v = luv_thresh(warped_im, LUV_THRESH)
    
    # combine all the thresholded images
    c = combined(l, b, v).astype('uint8')*255
    # find the lane lines
    if (first_frame | (len(left_fit) == 0) | (len(right_fit) == 0)):
        try: 
            left_fit, right_fit  = find_lane_pixel_windowing(c, fit_original_img=True, Minv=Minv,
                  **WINDOWING_LINED_DETECTION)
        except:
            return img, c, [], []
    
    else:
        try: 
            left_fit, right_fit = fit_using_previous_fit(c, left_fit, right_fit, fit_original_img=True, **PREVIOUS_FIT)
        except:
            try:
                left_fit, right_fit  = find_lane_pixel_windowing(c, fit_original_img=True, Minv=Minv,
                    **WINDOWING_LINED_DETECTION)
            except:
                left_fit, right_fit = [], []
    # draw the lines on the image
    try: 
        img = draw_lines(img, left_fit, right_fit)
    except:
        pass
    return img, c, left_fit, right_fit

# ------- #
# Results :
# ------- #
def preprocess_main():
    test_im = cv.imread('test_images/test4.jpg')
    test_im = cv.cvtColor(test_im, cv.COLOR_BGR2RGB)

    # Undistort image
    test_im = cv.undistort(test_im, mtx, dist, None, mtx)

    # Prespective Transform
    warped_im = cv.warpPerspective(test_im, M, IMAGE_SHAPE[::-1])

    s = Saturation_thresh(warped_im, S_THRESH)
    l = L_lab_thresh(warped_im, L_LAB_THRESH)
    b = B_lab_thresh(warped_im, B_LAB_THRESH)
    v = luv_thresh(warped_im, LUV_THRESH)
    c = combined(s, l, b, v)

    h = histogram(c)
    out_img = np.dstack((c, c, c))*255

    # Detect lane pixels using sliding window (for first time) on warped image
    left_fit, right_fit, windowing_result = find_lane_pixel_windowing(
        c, fit_original_img=False, **WINDOWING_LINED_DETECTION)
    
    # Detect lane pixels using sliding window (for first time) on original image
    windowing_left_fit_o, windowing_right_fit_o = find_lane_pixel_windowing(
        c, fit_original_img=True)

    # Detect lane pixels using previous fit for warped image
    result, left_fit, right_fit = fit_using_previous_fit(
        c, left_fit, right_fit, **PREVIOUS_FIT)

    # Detect lane pixels using previous fit (for original image)
    left_fit_o, right_fit_o = fit_using_previous_fit(
        c, left_fit, right_fit, fit_original_img=True)


    # -------------- #
    # Visualizations :
    # -------------- #
    # draw lines on the original image
    windowing_out_img = draw_lines(test_im, windowing_left_fit_o, windowing_right_fit_o)
    out_img = draw_lines(test_im, left_fit_o, right_fit_o)
    
    # Plot color spaces results: 
    plot_images([test_im, warped_im, s, l, b, v, c], 
                titles=["Original Image", "Warped Image", "S", "L", "B", "V", "Combined"],
                gridx=2, gridy=4, figsize=(20, 10))


    # Plot the results
    plot_images([test_im, windowing_result, windowing_out_img, result, out_img],
                ["Original image", "Windowing out image_warped", "Windowing out image_original",
                    "Lines with Prepspective Transformation", "Original image with fitted lines", " "],
                gridx=2, gridy=3, figsize=(20, 10))


if __name__ == "__main__":
    preprocess_main()