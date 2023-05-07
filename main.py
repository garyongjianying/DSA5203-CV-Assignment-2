import cv2
import numpy as np
import matplotlib.pyplot as plt


# supporting functions for imrect

def order_corners(corners):
    """
    Define a function to take in a set of 4 corner points (2D coordinates) and order them clockwise,
    starting from the top-left corner. Ordering is required to ensure correct perspective transformation
    when using cv2.getPerspectiveTransform() later.
    """
    # initialize an empty 4x2 numpy array to store the ordered corner points
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # sum the x and y coordinates for each corner point
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)] # represents corner with smallest sum of x and y coordinates - top left corner
    rect[2] = corners[np.argmax(s)] # represents corner with largest sum of x and y coordinates - bottom right corner
    
    # calculate the diffrence between x and y coordinates for each corner point
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)] # corner with smallest difference - top right corner
    rect[3] = corners[np.argmax(diff)] # corner with largest difference - bottom left corner

    return rect

def rectify_image(image, quadrilateral_points, rectified_size=(1200,1000), padding=100):
    """
    Takes an input image, the corner points of the found quadrilateral, the desired output size and default padding
    and returns an output 
    """
    # unpack the rectified default size into width and height
    width, height = rectified_size
    
    # define the output rectangle's corner points, with padding, since we do not just want to extract the object of interest
    output_rectangle_points = np.float32([[0 + padding, 0 + padding],
                                            [width - padding, 0 + padding],
                                            [width - padding, height - padding],
                                            [0 + padding, height - padding]])
    
    
    # calculate the perspective transform matrix M using the input quadrilateral corner points and output rectangle corner points
    # This is same as in the lecture notes for matrix H, and we use this to transform the input image into the rectified image.
    M = cv2.getPerspectiveTransform(np.float32(quadrilateral_points), output_rectangle_points)
    
    # Apply the perspective transform matrix M to the input image using warpPerspective
    rectified_image = cv2.warpPerspective(image, M, (width, height))
    
    # Returns the rectified image as a numpy array
    return rectified_image


def imrect(im1):
    """
    Takes in a numpy array for input image and returns the rectified image as a numpy array
    """
    ##################### Start of the main function for imrect function #############################
    image_copy = im1.copy() # copy the original image for visualization of contours / edge points detected
    
    # convert input image to gray scale
    gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    
    # Convert the floating-point grayscale image back to 8-bit unsigned integer
    gray_8bit = (gray * 255).astype(np.uint8)

    # apply a gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_8bit, (5, 5), 0)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # apply canny edge detection on blurred image
    edged = cv2.Canny(blurred, 75, 200)
    
    # since test1.jpg is a tricky image to detect with many broken lines in the contours, we need to do some morphological operations
    # perform morphological operations (dilation followed by erosion - closing) This operation helps to close small gaps and join broken edge lines.
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=3)
    closed = cv2.erode(dilated, kernel, iterations=3)
    
    # contours are found using findContours, returning a list of contours in the image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # original, does not detect test1.jpg well for the contours, hence we use morphological operations.
    # contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # the largest contour can be found using the MAX area, since the homework says the input image will always
    # have a prominent rectangular boundary in a monochrome background.
    largest_contour = max(contours, key=cv2.contourArea)
    
    # arcLength is used to find the perimeter of the largest contour found. Multiply by 0.02 to obtain epsilon,
    # which is used to control the approximation accuracy in approxPolyDP.
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    # approximates the contour shape. Result is a simplified contour that retains the general shape but with fewer vertices.
    # used to extract the 4 corner points
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    
    # If the largest contour has 4 corner points as in a rectangle, proceed with rectification
    # if not, raise exception and that the largest contour in the input image is not a quadrilateral.
    if len(approx_corners) == 4:
        # order the corner points
        ordered_corners = order_corners(approx_corners.reshape(4, 2))
        
        # Visualize the contour and corner points on the original image
        # draw contours on the image_copy, draws contours in place
        cv2.drawContours(image_copy, [approx_corners], -1, (0,255,0), 2)

        # draw the corner points on the image_copy
        for point in approx_corners:
            x, y = point[0]
            cv2.circle(image_copy, (x,y), 10, (255,0,0), -1)
            
        # use matplotlib to show the image with the contours and the corner points detected
        # Convert image_copy back to floating-point format with values in the range of 0 to 1
        # image_copy_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_copy_rgb.astype(np.float32)/ 255)
        image_copy_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        plt.imshow(image_copy_rgb)
        plt.title('Contours & Corner Points on original image')
        plt.axis('off')
        plt.show()
        
        # rectify the image with the order points, and desired output size
        rectified_image = rectify_image(im1, ordered_corners)
        
        # visualize the rectified image
        # rectified_image_float = rectified_image.astype(np.float32) / 255
        # rectified_image_rgb = cv2.cvtColor(rectified_image_float, cv2.COLOR_BGR2RGB)
        rectified_image_rgb = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rectified_image_rgb)
        plt.title('Rectified Image Output')
        plt.axis('off')
        plt.show()
        
        return rectified_image
    else:
        raise Exception("The largest contour is not a quadrilateral.")


if __name__ == "__main__":

    # This is the code for generating the rectified output
    # If you have any question about code, please send email to e0444157@u.nus.edu
    # fell free to modify any part of this code for convenience.
    img_names = ['./data/test1.jpg','./data/test2.jpg']
    for name in img_names:
        image = np.array(cv2.imread(name, -1), dtype=np.float32)/255.
        rectificated = imrect(image)
        cv2.imwrite('./data/Result_'+name[7:],np.uint8(np.clip(np.around(rectificated*255,decimals=0),0,255)))
