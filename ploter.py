import matplotlib.pyplot as plt
import cv2


def plot(bgr_image, pencil_sketch, mask, title, figsize=(15, 15)):
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    f.suptitle(title, size=20)
    ax1.imshow(rgb_image, cmap='gray')
    ax1.set_title('original image'.title())
    ax1.axis('off')
    ax2.imshow(pencil_sketch, cmap='gray')
    ax2.set_title('deep pencil sketch'.title())
    ax2.axis('off')
    ax3.imshow(mask, cmap='gray')
    ax3.set_title('sketch mask'.title())
    ax3.axis('off')
    plt.show()