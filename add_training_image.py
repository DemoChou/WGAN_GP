import cv2
import numpy as np
from tqdm import tqdm
import os

def rotate_without_blank(image, angle):

    # Get the image dimensions
    height, width = image.shape[:2]

    # Get the rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the bounding box of the rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    return rotated_image
def rotate_image_preserve(image, angle):
    # Get the image dimensions
    height, width = image.shape[:2]

    # Define the rotation center
    center = (width // 2, height // 2)

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the new dimensions of the rotated image
    new_width = int(height * np.abs(np.sin(np.radians(angle))) +
                    width * np.abs(np.cos(np.radians(angle))))
    new_height = int(width * np.abs(np.sin(np.radians(angle))) +
                     height * np.abs(np.cos(np.radians(angle))))

    # Adjust the rotation matrix for the new dimensions
    rotation_matrix[0, 2] += (new_width - width) // 2
    rotation_matrix[1, 2] += (new_height - height) // 2

    # Perform the rotation
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height))

    return rotated_image


def rotate_and_adjust_brightness(input_folder, output_folder, rotation_angle, brightness_factor):
    # Check if the output folder exists, create it if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Rotate the image
            rotated_image = rotate_without_blank(image, rotation_angle)
            # (height, width) = image.shape[:2]
            # rotation_matrix = cv2.getRotationMatrix2D(
            #     (width/2, height/2), rotation_angle, 1.0)
            # rotated_image = cv2.warpAffine(
            #     image, rotation_matrix, (width, height))

            # Adjust the brightness
            adjusted_image = cv2.convertScaleAbs(
                rotated_image, alpha=brightness_factor, beta=0)

            # Save the processed image
            output_path = os.path.join(
                output_folder, str(rotation_angle)+"_"+str(brightness_factor)+"_"+filename)
            cv2.imwrite(output_path, adjusted_image)

            # print(f"Image {filename} processed!")


# Test
input_folder = 'C:\\Users\\APICS\\Desktop\\Chun\\dataset'  # Input folder path
output_folder = 'C:\\Users\\APICS\\Desktop\\Chun\\dataset_add'  # Output folder path
brightness_factor = [-1.5, -1, -0.5, 0.5, 1, 1.5]
# rotation_angle = 90  # Rotation angle (degrees)
# brightness_factor = 1.5  # Brightness adjustment factor
for rotation_angle in tqdm(np.linspace(start=0, stop=90, num=10)):
    for b in range(6):
        rotate_and_adjust_brightness(
            input_folder, output_folder, rotation_angle, brightness_factor[b])
