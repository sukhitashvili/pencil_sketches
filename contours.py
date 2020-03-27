import cv2
from sketches import pencil_sketch

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))


def render(img_rgb, weight=29):
    img_height = img_rgb.shape[0]
    img_width = img_rgb.shape[1]
    if not (weight >= 3):
        raise RuntimeError('parameter weight must be no less than 3!')
    if not weight < min(img_height, img_width):
        raise RuntimeError('parameter weight must be less than min(your_image_height, your_image_width)!')
    if not (weight % 2) == 1:  # if not odd number then make it odd
        weight -= 1
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (weight, weight), 0, 0)
    img_blend = cv2.divide(img_gray, img_blur, scale=256)
    return img_blend


def deep_contour(img_, weight=29, dilate=0):
    img = render(img_, weight=weight)
    img_deep = pencil_sketch(img)
    img = (255 - img_deep).astype('uint8')
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if dilate:
        img = cv2.dilate(img, kernel, iterations=dilate)
        img_deep = ((img == 0) * 255).astype('uint8')
        img = cv2.medianBlur(img, 3)
        img_deep = cv2.medianBlur(img_deep, 5)
    return img_deep, img


if __name__ == '__main__':
    my_img = cv2.imread('./images/person3.jpg')  # if one channel image is given then cv reads image copying its channel to other 2 channels, so making image rgb again!
    deep_img, img = deep_contour(my_img, weight=29, dilate=0)
    cv2.imshow('my image', my_img)
    cv2.imshow('deep pencil', deep_img)
    cv2.imshow('binary image', img)
    # cv2.imwrite('deep_contour.jpg', img)

    cv2.waitKey()
    cv2.destroyAllWindows()
