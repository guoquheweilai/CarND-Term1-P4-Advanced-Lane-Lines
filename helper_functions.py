import numpy as np
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
	def __init__(self):
		# First run?
		self.first_run = True
		# was the line detected in the last iteration?
		self.detected = False
		# How many times of losing lines in a row?
		self.missing_land_lines_counter = 0
		# x values of the last n fits of the line
		self.recent_xfitted = [] 
		#average x values of the fitted line over the last n iterations
		self.bestx = None
		# polynomial coefficients values of the last n fits of the line
		self.recent_fit = None
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = None  
		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]  
		#radius of curvature of the line in some units
		self.radius_of_curvature = None 
		#distance in meters of vehicle center from the line
		self.line_base_pos = None 
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float') 
		#x values for detected line pixels
		self.allx = None  
		#y values for detected line pixels
		self.ally = None

# Create land lines for left land line and right land line
# The name is hard coded, please don't change them
left_land_line = Line()
right_land_line = Line()
img_size = (1280, 720)

# Define source points coordinate
src_coordins = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 15), img_size[1]],
    [(img_size[0] * 5 / 6) + 45, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
# Define destination points coordinate
dst_coordins = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
	
def convert_BGR2GRAY(img):
    retimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return retimg

def convert_BGR2RGB(img):
    retimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return retimg

def abs_sobel_threshold(img, orient='x', grad_thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = convert_BGR2GRAY(img)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if 'x' == orient:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif 'y' == orient:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        sobel = 0
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > grad_thresh[0]) & (scaled_sobel < grad_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    binary_output = sxbinary
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = convert_BGR2GRAY(img)
    # 2) Take the gradient in x and y separately
    # Gradient in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    # Gradient in y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(sobelx**2+sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(abs_sobelxy)/255
    scaled_sobelxy = np.uint8(abs_sobelxy/scale_factor)
    # 5) Create a binary mask where mag thresholds are met
    #sxybinary = np.zeros_like(scaled_sobelxy)
    sxybinary = np.zeros_like(scaled_sobelxy)
    sxybinary[(scaled_sobelxy > mag_thresh[0])&(scaled_sobelxy < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
#     binary_output = np.copy(img) # Remove this line
    binary_output = sxybinary
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = convert_BGR2GRAY(img)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    graddir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(graddir)
    binary_output[(graddir > dir_thresh[0])&(graddir < dir_thresh[1])]=1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output

def gray_threshold(img, thresh=(20, 100)):
    # Using cv2.imread so need to use cv2.COLOR_BGR2GRAY
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_binary = np.zeros_like(gray_img)
    gray_binary[(gray_img > thresh[0]) & (gray_img <= thresh[1])] = 1
    return gray_img, gray_binary

def hls_threshold(img, color='h', color_thresh=(20, 100)):
    # Using cv2.imread so need to use cv2.COLOR_BGR2HLS
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if color == 'h':
        color_img = hls_img[:,:,0]
    elif color == 'l':
        color_img = hls_img[:,:,1]
    elif color == 's':
        color_img = hls_img[:,:,2]
    else:
        color_img = hls_img[:,:,:]
    color_binary = np.zeros_like(color_img)
    color_binary[(color_img > color_thresh[0]) & (color_img <= color_thresh[1])] = 1
    return color_img, color_binary

def rgb_threshold(img, color='r', color_thresh=(20, 100)):
        # Using cv2.imread so need to use cv2.COLOR_BGR2RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if color == 'r':
        color_img = rgb_img[:,:,0]
    elif color == 'g':
        color_img = rgb_img[:,:,1]
    elif color == 'b':
        color_img = rgb_img[:,:,2]
    else:
        color_img = rgb_img[:,:,:]
    color_binary = np.zeros_like(color_img)
    color_binary[(color_img > color_thresh[0]) & (color_img <= color_thresh[1])] = 1
    return color_img, color_binary

def Perspective_Transfer_Warper(img, src_coordins, dst_coordins):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src_coordins, dst_coordins)
    ret_Img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return ret_Img
	
def find_land_line_pixels(out_img, binary_img, list_lane_line_pixel_coords, nwindows, x_base, margin, minpix):
    # Set height of windows
    window_height = np.int(binary_img.shape[0]/nwindows)
	
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_img.shape[0] - (window+1)*window_height
        win_y_high = binary_img.shape[0] - window*window_height
        win_xleft_low = x_base - margin
        win_xleft_high = x_base + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        # Append these indices to the lists
        list_lane_line_pixel_coords.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_base = np.int(np.mean(nonzerox[good_inds]))
    # Concatenate the arrays of indices
    list_lane_line_pixel_coords = np.concatenate(list_lane_line_pixel_coords)
    return list_lane_line_pixel_coords

def fit_polynomial(binary_img, list_lane_line_pixel_coords, poly_order=2):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
	
	# Extract left and right line pixel positions
    x = nonzerox[list_lane_line_pixel_coords]
    y = nonzeroy[list_lane_line_pixel_coords]
    # Fit a second order polynomial to each
    ret_fit_polynomial = np.polyfit(y, x, poly_order)
    return ret_fit_polynomial
	
def process_image(img):
	# Initialization
	#global left_land_line
	#global right_land_line

	# Run functions
	gradx_binary = abs_sobel_threshold(img, orient='x', grad_thresh=(20, 100))
	s_img, s_binary = hls_threshold(img, color='s', color_thresh=(90, 255))

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(gradx_binary)
	combined_binary[(s_binary == 1) | (gradx_binary == 1)] = 1

	# Compute and apply perspective transform
	Warper = Perspective_Transfer_Warper(combined_binary, src_coordins, dst_coordins)

	# Take a histogram of the bottom half of the image
	histogram = np.sum(Warper[Warper.shape[0]//2:,:], axis=0)

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((Warper, Warper, Warper))*255

	# Choose the number of sliding windows
	nwindows = 20
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = Warper.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Find pixels coordinates in each land lines
	list_x_left_coords = find_land_line_pixels(out_img, Warper, left_lane_inds, nwindows, leftx_base, margin, minpix)
	list_x_right_coords = find_land_line_pixels(out_img, Warper, right_lane_inds, nwindows, rightx_base, margin, minpix)

	# Fit a second order polynomial to each land lines
	fit_poly_x_left = fit_polynomial(Warper, list_x_left_coords, poly_order=2)
	fit_poly_x_right = fit_polynomial(Warper, list_x_right_coords, poly_order=2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, Warper.shape[0]-1, Warper.shape[0] )
	plotx_left = fit_poly_x_left[0]*ploty**2 + fit_poly_x_left[1]*ploty + fit_poly_x_left[2]
	plotx_right = fit_poly_x_right[0]*ploty**2 + fit_poly_x_right[1]*ploty + fit_poly_x_right[2]

	# Define y-value where we want radius of curvature
	# I'll choose the mean y-value, corresponding to the middle of the image
	y_eval = np.mean(ploty)

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, plotx_left*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, plotx_right*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	# Caluculate the bottom points coodinate
	bottom_y = np.max(ploty)
	bottom_x_left = fit_poly_x_left[0]*bottom_y**2 + fit_poly_x_left[1]*bottom_y + fit_poly_x_left[2]
	bottom_x_right = fit_poly_x_right[0]*bottom_y**2 + fit_poly_x_right[1]*bottom_y + fit_poly_x_right[2]
	# Caluculate the offset from the center of the lane
	offset_center_pix = (bottom_x_left + bottom_x_right)/2 - img.shape[1]/2
	offset_center_m = offset_center_pix*xm_per_pix

	# Smooth over the last n frames of video
	# Compute the inverse perspective transform
	Minv = cv2.getPerspectiveTransform(dst_coordins, src_coordins)
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(Warper).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([plotx_left, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([plotx_right, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
	# Add real-time analysis data to the image
	font = cv2.FONT_HERSHEY_SIMPLEX
	result = cv2.putText(result,'Radius of Curvature = %s(m)' % (left_curverad),(20,50), font, 1.5,(255,255,255),2,cv2.LINE_AA)
	result = cv2.putText(result,'Vehicle is %s(m) left of center' % (offset_center_m),(20,90), font, 1.5,(255,255,255),2,cv2.LINE_AA)
		
		
	return result