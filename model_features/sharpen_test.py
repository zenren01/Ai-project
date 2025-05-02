import cv2
def sharpen_image():
    # Load the image
    image_path = input()
    image = cv2.imread(image_path)
    image = cv2.resize(image, (700, 700))

    # Apply detail enhancement to sharpen the image
    sharpened = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

    # Display the original image and the sharpened image
    cv2.imshow('Original', image)
    cv2.imshow('Sharpened', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function to start the image sharpening process
sharpen_image()

