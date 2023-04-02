import cv2
import numpy as np

def main():
    img = cv2.imread('data/sample_score_sheet.jpg', -1)

    # clip paper
    x_size = 2000
    y_size = int(x_size * 1.412)  # Assume the paper is B5 size
    img = clipped_paper(img, x_size, y_size)
    cv2.imwrite("data/clipped_paper.png", img)

    # recognize notations
    notations = list_notations(img)
    print('notations:', notations)

    cv2.imshow('window', img)
    cv2.waitKey(5000)


def clipped_paper(img, x_size, y_size):
    """
    Return image of the paper, which is clipped from the argument image

    Args:
        img: source image
        x_size: output image x size
        y_size: output image y size
    """

    # Detect corners of the paper
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_th1 = cv2.threshold(img_gray, 220, 255, cv2.THRESH_TOZERO_INV)
    img_not = cv2.bitwise_not(img_th1)
    ret, img_th2 = cv2.threshold(img_not, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_th2, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find paper contour
    picture_outline = {'contour': None, 'size': 0}  # should be max size
    paper_outline = {'contour': None, 'size': 0}  # should be second max size
    for con in contours:
        size = cv2.contourArea(con)
        if size >= picture_outline['size']:
            picture_outline['contour'] = con
            picture_outline['size'] = size
            continue
        if size >= paper_outline['size']:
            paper_outline['contour'] = con
            paper_outline['size'] = size
            continue

    # reshape paper contour to the original size ratio
    epsilon = 0.1*cv2.arcLength(paper_outline['contour'], True)
    paper_corners = cv2.approxPolyDP(paper_outline['contour'], epsilon, True)
    reshaped_con = np.array([[[0, 0]], [[x_size, 0]],
                            [[x_size, y_size]], [[0, y_size]]],
                            dtype="int32")
    reshape_M = cv2.getPerspectiveTransform(np.float32(paper_corners),
                                            np.float32(reshaped_con))
    img_trans = cv2.warpPerspective(img, reshape_M, (x_size, y_size))

    return img_trans


def list_notations(img):
    return []



if __name__ == '__main__':
    main()
