import cv2

def select_and_clip_image():
    # Global variables
    ref_point = []
    cropping = False

    # Mouse callback function for selecting the region
    def click_and_crop(event, x, y, flags, param):
        nonlocal ref_point, cropping

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]
            cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            cropping = False

            # Draw a rectangle around the selected region
            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("Original Image", image)

    # Prompt the user for the image file path
    image_path = input("Enter the image file path: ")

    # Load the original image
    image = cv2.imread(image_path)

    # Create a window and bind the mouse callback function to it
    cv2.namedWindow("Original Image")
    cv2.setMouseCallback("Original Image", click_and_crop)

    clipped_image = None

    while True:
        clone = image.copy()
        if len(ref_point) == 2:
            cv2.rectangle(clone, ref_point[0], ref_point[1], (0, 255, 0), 2)

        cv2.resize(clone, (700, 700))
        cv2.imshow("Original Image", clone)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' key to reset the region
        if key == ord("r"):
            image = cv2.imread(image_path)
            clipped_image = None

        # Press 'p' key to display the clipped image
        elif key == ord("p"):
            if len(ref_point) == 2:
                x_start, y_start = ref_point[0]
                x_end, y_end = ref_point[1]
                x_min, x_max = min(x_start, x_end), max(x_start, x_end)
                y_min, y_max = min(y_start, y_end), max(y_start, y_end)

                clipped_image = image[y_min:y_max, x_min:x_max]
                cv2.imshow("Clipped Image", clipped_image)

        # Press 'Esc' key to exit the program
        elif key == 27:
            break

    cv2.destroyAllWindows()