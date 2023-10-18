
# (c) Alex Spirin 2023

import cv2
import numpy as np
from skimage.morphology import disk, binary_erosion, binary_dilation
from torchvision.utils import flow_to_image as flow_to_image_torch

def extract_occlusion_mask(flow, threshold=10):
    flow = flow.clone()[0].permute(1, 2, 0).detach().cpu().numpy()
    h, w = flow.shape[:2]

    """
    Extract a mask containing all the points that have no origin in frame one.

    Parameters:
        motion_vector (numpy.ndarray): A 2D array of motion vectors.
        threshold (int): The threshold value for the magnitude of the motion vector.

    Returns:
        numpy.ndarray: The occlusion mask.
    """
    # Compute the magnitude of the motion vector.
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold the magnitude to identify occlusions.
    occlusion_mask = (mag > threshold).astype(np.uint8)

    return occlusion_mask, mag

def edge_detector(image, threshold=0.5, edge_width=1):
    """
    Detect edges in an image with adjustable edge width.

    Parameters:
        image (numpy.ndarray): The input image.
        edge_width (int): The width of the edges to detect.

    Returns:
        numpy.ndarray: The edge image.
    """
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Sobel edge map.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=edge_width)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=edge_width)

    # Compute the edge magnitude.
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Normalize the magnitude to the range [0, 1].
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    # Threshold the magnitude to create a binary edge image.

    edge_image = (mag > threshold).astype(np.uint8) * 255

    return edge_image

def get_unreliable(flow):
    # Mask pixels that have no source and will be taken from frame1, to remove trails and ghosting.

    # flow = flow[0].cpu().numpy().transpose(1,2,0)

    # Calculate the coordinates of pixels in the new frame
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = x + flow[..., 0]
    new_y = y + flow[..., 1]

    # Create a mask for the valid pixels in the new frame
    mask = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)

    # Create the new frame by interpolating the pixel values using the calculated coordinates
    new_frame = np.zeros((flow.shape[0], flow.shape[1], 3))*1.-1
    new_frame[new_y[mask].astype(np.int32), new_x[mask].astype(np.int32)] = 255

    # Keep masked area, discard the image.
    new_frame = new_frame==-1
    return new_frame, mask

def remove_small_holes(mask, min_size=50):
    # Copy the input binary mask
    result = mask.copy()

    # Find contours of connected components in the binary image
    contours, hierarchy = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour
    for i in range(len(contours)):
        # Compute the area of the i-th contour
        area = cv2.contourArea(contours[i])

        # Check if the area of the i-th contour is smaller than min_size
        if area < min_size:
            # Draw a filled contour over the i-th contour region
            cv2.drawContours(result, [contours[i]], 0, 255, -1, cv2.LINE_AA, hierarchy, 0)

    return result

def filter_unreliable(mask, dilation=1):
  img = 255-remove_small_holes((1-mask[...,0].astype('uint8'))*255, 200)
  img = binary_erosion(img, disk(1))
  img = binary_dilation(img, disk(dilation))
  return img

def make_cc_map(predicted_flows, predicted_flows_bwd, dilation=1, edge_width=11, flow_to_image=flow_to_image_torch):
  flow_imgs = flow_to_image(predicted_flows_bwd)
  edge = edge_detector(flow_imgs.astype('uint8'), threshold=0.1, edge_width=edge_width)
  res, _ = get_unreliable(predicted_flows)
  _, overshoot = get_unreliable(predicted_flows_bwd)
  joint_mask = np.ones_like(res)*255
  joint_mask[...,0] = 255-(filter_unreliable(res, dilation)*255)
  joint_mask[...,1] = (overshoot*255)
  joint_mask[...,2] = 255-edge

  return joint_mask
