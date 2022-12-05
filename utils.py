import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

##############################################
############ Configurations ##################
FIGSIZE = 20, 12
FONTSIZE = 20
LINETHICKNESS = 5
##############################################


def draw_image( img:np.ndarray=None,
                img_path:str=None,
                title:str=None,
                figsize:tuple=FIGSIZE, 
                fontsize=FONTSIZE):
    '''
    drawing image in form of 
    
    Args: 
        img:        image to draw.
        img_path:   path to image to draw
    '''
    if img_path is not None: 
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.xticks([]), plt.yticks([]); plt.title(title, fontsize=fontsize)
    cmap = 'gray' if img.ndim==2 else None
    plt.imshow(img)

def image_grid( rows:int, cols:int, images:list,
                titles:list=None,
                figsize=FIGSIZE,
                fontsize=FONTSIZE):
    '''
    Args: 
        rows: number of figures in each row.
        cols: number of figures in each col.
        images: list of images to draw.

    Raises: 
        OverSuppliedImages: if number of images can't fit in the supposed grid.
        TitlesDoNotMatches: raises exception when number of images doesn't equal number of titles 
    '''

    # Exceptions
    if rows*cols < len(images):
        raise Exception("OverSuppliedImages: Number of Images can not fit in image grid")
    if titles is None: 
        titles = [''] * len(images)
    else:
        if (len(titles)!=len(images)):
           raise Exception(f"TitlesDoNotMatches: Can not assign Titles with length: {len(titles)} to Images with length: {len(images)}") 

    _, axs = plt.subplots(rows, cols, figsize=FIGSIZE)
    for i, ax in enumerate(axs.flatten()):
        ax.set_xticks([]), ax.set_yticks([]); ax.set_title(titles[i], fontsize=fontsize)

        cmap = 'gray' if images[i].ndim==2 else None
        ax.imshow(images[i], cmap=cmap)


def draw_lines_rho_theta(image:np.ndarray, lines:list, thickness=LINETHICKNESS):
    '''
    Args: 
        image: image to draw lines on.
        lines: list of tuples where each tuple carries rho and theta of line.
    
    Returns: 
        image: image after drawing lines on.
    '''
    img = image.copy()
    for line in lines:
        # line equation: x = r * cos(theta), y = r * sin(theta)
        r, theta = line[0]
        
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        
        # find two points in this line
        pt1 = int(x0 - 1000 * (- np.sin(theta))), int(y0 - 1000 * (np.cos(theta)))
        pt2 = int(x0 + 1000 * (- np.sin(theta))), int(y0 + 1000 * (np.cos(theta)))
        
        cv.line(img, pt1, pt2, (255, 0, 0), thickness)
    
        
    return img
    
def draw_lines_end_points(image:np.ndarray, lines:list, thickness=LINETHICKNESS):
    '''
    Args: 
        image: image to draw lines on.
        lines: list of lists 
        , each list contains [x0, y0, x1, y1] coordinates of end points in line.
        thickness: thickness of line.

    Returns:
        image: image after drawing lines on. 
    '''
    img = image.copy()
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(img,(x1,y1),(x2,y2),(255,0,0),thickness)

    return img