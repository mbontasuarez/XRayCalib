import cv2
import numpy as np
import itertools as itr
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
from pathlib import Path


class XRaySeg:
    """
    Class that handles X-ray image segmentation of phantom fiducial markers.
    """
    def __init__(self, file_name):
        """
        Initialize XRaySeg object and loads X-Ray image from file_name.
        Args:
            file_name (str): name and path to X-ray image (.jpg).
        """

        # Initialize parameters
        self.image_segmented = False            # Has the image been segmented?
        self.image_contoured = False            # Has the image been contoured?

        self.num_markers_found = 0              # Number of fiducial markers found through segmentation
        self.detected_circles = []              # (x,y) coordinates and radius of the fiducial markers found
        self.labels = []                        # Segmented markers true label
        self.contoured_image = []               # Image with the found fiducial markers contoured

        #Load X-ray image from the file_name
        self.X = cv2.imread(file_name, cv2.IMREAD_COLOR)

        # Converts image to grayscale
        self.grImage = cv2.cvtColor(self.X, cv2.COLOR_BGR2GRAY)

    def segment(self,minDist = 5, minR = 1, maxR = 10, prm1 = 30, prm2 = 15):
        """
        Segments fiducial markers in the X-ray image using Hough Transform and randomly assigns a label.
        Args:
            minDist (int): minimum distance between the centre of identified circles (pixel size).
            rMin (int): minimum radius of the segmented fiducials (pixel size).
            rMax (int): maximum radius of the segmented fiducials (pixel size).
        """

        # Blur image using a 3 x 3 kernel
        grImage_blurred = cv2.blur(self.grImage, (3, 3))

        # Apply Hough transform on the blurred image. Returns a (Nx3) matrix where N is the number of circles detected
        # and the 2nd dimmension is (horizontal pixel position, vertical index position, radius in pixel)
        detected_circles = cv2.HoughCircles(grImage_blurred,
                                            cv2.HOUGH_GRADIENT, 1, minDist,
                                            param1=prm1,
                                            param2=prm2,
                                            minRadius=minR,
                                            maxRadius=maxR)

        if detected_circles is not None:
            # Make them integers to be consistent with index positions
            detected_circles = np.uint16(np.around(detected_circles[0]))
            detected_circles = np.array(detected_circles.tolist())        # See how to fix this

            self.detected_circles = detected_circles                      # Save the markers coordinates
            self.num_markers_found = self.detected_circles[:, 0].size     # Save the number of markers found
            self.labels = np.array(range(self.num_markers_found))         # Provide aletory labels

            print(f"Number of fiducial markers detected: {self.num_markers_found}")

            self.image_segmented = True

            return self.detected_circles

        else:
            print("No fiducial markers found. Consider the parameters entered.")
            self.image_segmented = False
            return False

    def segment_search(self, num_markers, step_size_rmax = 1, step_size_prm2 = 1, max_iter = 10, verbose = True):
        """
        NOT FINISHED BUT FUNCTIONAL
        Segments the fiducial markers in the X-ray image using Hough Transform by automatically searching for the best
        parameters based on the number of markers desired.
        Args:
            minDist (int): minimum distance between the centre of identified circles (pixel size).
            rMin (int): minimum radius of the segmented fiducials (pixel size).
            rMax (int): maximum radius of the segmented fiducials (pixel size).
        """

        # Blur image using a 3 x 3 kernel
        valid_segmentation = False
        grImage_blurred = cv2.blur(self.grImage, (3, 3))


        #Initial values
        rMax_search = 3
        prm2_search = 15

        # Apply Hough transform on the blurred image. Returns a (Nx3) matrix where N is the number of circles detected
        # and the 2nd dimmension is (horizontal pixel position, vertical index position, radius in pixel)
        iter = 0
        prev_state = 0
        while (not valid_segmentation) and (iter < max_iter):
            if verbose:
                print(f"Iteration {iter}: Rmax = {rMax_search}, Prm2 = {prm2_search}")
            M = 0
            detected_circles = cv2.HoughCircles(grImage_blurred,
                                                cv2.HOUGH_GRADIENT, 1, 10,
                                                param1=30,
                                                param2=prm2_search,
                                                minRadius=1,
                                                maxRadius=rMax_search)

            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles[0]))

                M = detected_circles[:,0].size
                if M == num_markers:
                    valid_segmentation = True
                    iter = max_iter
                    if verbose:
                        print(f"\tValid number of fiducial markers found: {M}.")
                    break
                elif M > num_markers:
                    if verbose:
                        print(f"\tNumber of markers found is too high: {M}")
                elif M < num_markers:
                    if verbose:
                        print(f"\tNumber of markers found is too low: {M}")
                    else:
                        print("ERR01: segmentation search base case reached.")
            else:
                if prev_state < num_markers:
                    rMax_search += 1
                if verbose:
                    print("\tNo markers found.")

            if iter == 0:
                prev_state = M
            iter += 1
        if valid_segmentation:
            # Make them integers to be consistent with index positions
            detected_circles = np.array(detected_circles.tolist())        # See how to fix this
            self.detected_circles = detected_circles                      # Save the markers coordinates
            self.num_markers_found = self.detected_circles[:, 0].size     # Save the number of markers found
            self.labels = np.array(range(self.num_markers_found))         # Provide random labels

            if verbose:
                print(f"Number of fiducial markers detected: {self.num_markers_found}")

            self.image_segmented = True

            return self.detected_circles

        else:
            print("No fiducial markers found. Consider the parameters entered.")
            self.image_segmented = False
            return False

    def segment_matrix(self,minR,maxR,minPrm2,maxPrm2):
        """
               NOT FINISHED BUT FUNCTIONAL
               Segments the fiducial markers in the X-ray image using Hough Transform by automatically searching for the best
               parameters based on the number of markers desired.
               Args:
                   minDist (int): minimum distance between the centre of identified circles (pixel size).
                   rMin (int): minimum radius of the segmented fiducials (pixel size).
                   rMax (int): maximum radius of the segmented fiducials (pixel size).
               """

        R_search = np.array(range(minR,maxR))
        Prm2_search = np.array(range(minPrm2,maxPrm2))

        matrix_results = np.zeros((Prm2_search.shape[0],R_search.shape[0]))

        grImage_blurred = cv2.blur(self.grImage, (3, 3))

        for i, Prm2 in enumerate(Prm2_search):
            for j, r in enumerate(R_search):
                M = 0
                detected_circles = cv2.HoughCircles(grImage_blurred,
                                                    cv2.HOUGH_GRADIENT, 1, 10,
                                                    param1=30,
                                                    param2=Prm2,
                                                    minRadius=1,
                                                    maxRadius=r)

                if detected_circles is not None:
                    detected_circles = np.uint16(np.around(detected_circles[0]))
                    M = detected_circles[:, 0].size
                    #print(f"Markers found: {M}")
                else:
                    #print(f"No markers found.")
                    a = 3 + 3
                matrix_results[i, j] = M


        matrix_results[matrix_results != 16] = 0
        color_map = plt.imshow(matrix_results)
        color_map.set_cmap("jet")
        plt.colorbar()


        print(f"Minimum radius: {}")
        return matrix_results

    def contour(self):
        """
            Draws a contour around the segmented circles at the found radius.
        """
        if self.image_segmented:
            image_temp = self.X.copy()
            for p in self.detected_circles:
                x, y, r = p

                # Draw circumference of the circle.
                cv2.circle(image_temp, (x, y), r, (0, 230, 255), 2)

                # Draw the centre
                cv2.circle(image_temp, (x, y), 1, (10, 87, 125), 2)

            self.image_contoured = True

            # Following line overlays transparent rectangle over the image
            self.contoured_image = image_temp
        else:
            print("Segment the image first.")
            self.image_contoured = False

    def visualize(self, save_image = True):
        """
            Allows for the visualization of the segmented fiducial markers. If the image has not been contoured then
            this function will simply display the input image.
        """
        fig, ax = plt.subplots()

        if self.image_contoured:
            ax.imshow(self.contoured_image)

            for i in range(self.num_markers_found):
                xy = self.detected_circles[i, 0:2]
                plt.annotate(f"{int(self.labels[i])}", xy=xy, size=10, fontweight='bold')
            plt.title(f"Segmented fiducial markers: {self.num_markers_found}")
            if save_image:
                fig.savefig("results/contoured_image_" + str(datetime.today().strftime("%y-%m-%d"))+".png",
                            bbox_inches="tight", pad_inches=0)
        else:
            print(self.X.shape)
            ax.imshow(self.X)
            plt.title("Input image")
            print("Note: the image has not been contoured.")

    def visualize_sets(self, pt_array):
        """
            Allows for the visualization of specific points.
            arg:
                pt_array ((M,3) array): index position of the points you wish to visualize
        """
        color_map = list((cm.gist_rainbow(range(0,2000,40))*255)[:,0:3])

        image_temp = self.contoured_image.copy()
        M = pt_array.shape[0]

        for indx, pi_coll in enumerate(pt_array):              # Extract a set of 3 points
            for p in pi_coll:
                x, y, r = self.detected_circles[p]
                R, G, B = color_map[indx]

                # Draw circumference of the circle.
                cv2.circle(image_temp, (x, y), r, (R, G, B), 2)

                # Draw the centre
                cv2.circle(image_temp, (x, y), 1, (10, 87, 125), 2)

        fig, ax = plt.subplots()
        ax.imshow(image_temp)
        r = np.unique(pt_array)

        for i in range(self.num_markers_found):
            xy = self.detected_circles[i,0:2]
            plt.annotate(f"{int(self.labels[i])}", xy=xy, size=10, fontweight='bold')
        plt.title("Visualization of collinear sets")
        fig.savefig("results/contoured_sets_" + str(datetime.today().strftime("%y-%m-%d")) + ".png",
                    bbox_inches="tight", pad_inches=0)


class CCalibration:

    def __init__(self):
        self.train_image = []
        self.test_image = []
        self.total_markers = 0
        self.pt_original_coordinates = []
        self.pt_train_coordinates = []
        self.train_image_loaded = False
        self.coll_sets = []
        self.DLT_params = []

    def set_parameters(self, coordinates, coll_threshold=50, max_iter_seg=10):
        """
        Handles the parameters needed for the computation of the DLT
        Args:
            coordinates (numpy array of size (n,3) where n is the number of fiducial markers): List of (x,y,z)
                        coordinates of all fiducial markers (mm).
            coll_threshold (int): threshold used to determine if points are collinear
            max_iter_seg (int): max number of iteration for the automatic fiducial segmentation
        """
        self.total_markers = coordinates[:, 0].size             # Extract the total number of available fiducials
        self.pt_original_coordinates = coordinates              # Store original 3D coordinates
        self.coll_thrsh = coll_threshold                        # Store threshold used to determine collinearity
        self.max_iter_seg = max_iter_seg                        # Max iterations to find optimal fiducial segmentation

        #Compute and store a distance matrix for all original 3D coordinate points
        self.pt_original_distances = self.compute_distances(self.pt_original_coordinates)

    def extract_fiducials(self, file_name, visualize = False):
        """
        Function that handles the segmentation of the training image.
        Args:
            file_name (str): name and path of X-ray image (.jpg).
            visualize (bool): allows the visualization of the segmented fiducial markers found.
        Returns:
            pt_train_coordinates (numpy array (n,2) where n is the number of points found): (x,y) pixel positions of
            identified fiducial markers.
        """
        print("Loading and segmenting training image.")
        self.train_image = XRaySeg(file_name)                                             # Load image
        tmp_coordinates = self.train_image.segment(minDist=5, minR=1, maxR=10, prm1=30, prm2=15)    # Segment image
        # CHANGE FOR AUTOMATIC SEGMENTATION

        if tmp_coordinates is not False:
            self.pt_train_coordinates = tmp_coordinates[:, 0:2]                            # Extract (x,y) coordinates
            print(f"\t Image segmented correctly, {self.train_image.num_markers_found} fiducial markers found.")

            # Compute and store a distance matrix for all original 2D coordinate points
            self.pt_train_distances = self.compute_distances(self.pt_train_coordinates)
        else:
            print(f"\t Segmentation failed.")
            return 0

        if visualize:
            self.train_image.contour()
            self.train_image.visualize()

        return self.pt_train_coordinates

    def label_fiducials(self):
        """
        Automatically associates the segmented fiducial marker with their corresponding label
        Args:
            None
        """
        # ADD CHECKS TO SEE IMAGE IS PROPERLY SEGMENTED

        # Organize distances from smallest to biggest
        I = np.around(np.sort(self.pt_train_distances, axis=0))   #Sort pixel fiducial distances from smallest to biggest
        R = np.sort(self.pt_original_distances, axis=0)           #Sort real fiducial distances from smallest to biggest

        # Normalize
        Ic = (I - np.mean(I, axis=0))/np.std(I, axis=0)
        Rc = (R - np.mean(R, axis=0))/np.std(R, axis=0)

        M = Ic.shape[1]
        new_labels = np.zeros((1, M))

        def corr(x,y):
            return np.sum(x*y)/np.sqrt(np.sum(x**2)*np.sum(y**2))

        for i in range(M):
            tmp = np.array([corr(Ic[:, i],Rc[:, j]) for j in range(self.total_markers)])
            new_labels[0,i] = np.argmax(tmp)

        return new_labels[0,:].T

    def label_fiducials_adv(self):
        """
        Automatically associates the segmented fiducial marker with their corresponding label
        Args:
            None
        """
        # ADD CHECKS TO SEE IMAGE IS PROPERLY SEGMENTED

        # Organize distances from smallest to biggest
        I = np.around(np.sort(self.pt_train_distances, axis=0))   #Sort pixel fiducial distances from smallest to biggest
        R = np.sort(self.pt_original_distances, axis=0)           #Sort real fiducial distances from smallest to biggest

        # Normalize
        #Ic = (I - np.mean(I, axis=0))/np.std(I, axis=0)
        #Rc = (R - np.mean(R, axis=0))/np.std(R, axis=0)
        Ic = I
        Rc = R

        M = Ic.shape[1]
        new_labels = np.zeros((1, M))

        def corr(x,y):
            return np.sum(x*y)/np.sqrt(np.sum(x**2)*np.sum(y**2))

        for i in range(M):
            tmp = np.array([corr(Ic[:, i],Rc[:, j]) for j in range(self.total_markers)])
            new_labels[0,i] = np.argmax(tmp)

        return new_labels[0,:].T

    def manual_labelling(self):
        """
        Manual association of the segmented fiducial marker with their corresponding label
        Args:
            None
        """
        new_labels = self.train_image.labels

        finished_labelling = False
        while not finished_labelling:
            a = input("Modify a point? [y/n]")
            if a == "y":
                old_tmp = input("Enter array of points you want to change separated by a , (e.g. 1,2,4): ")
                old = old_tmp.split(",")
                new_tmp = input("Enter the new label you want to assign to the above points (e.g. 0,4,9): ")
                new = new_tmp.split(",")

                for dx, j in enumerate(old):
                    new_labels[new_labels == int(j)] = int(new[dx])
            elif a == "n":
                finished_labelling = True

        self.set_labels(new_labels)

    def set_labels(self, new_labels, visualize=False):
        self.train_image.labels = new_labels
        if visualize:
            self.train_image.visualize()

    def get_labels(self):
        return self.train_image.labels

    def compute_distances(self,pt_array):
        """
        Computes the euclidean distance between every point.
        Args:
            pt_array (list (n,k) where n is the number of points and k is the dimension (2D or 3D)): contains
            the (x,y) or (x,y,z) coordinates of the points.
        Returns:
            D (numpy array (n,n)): contains the distances between every point.
        """
        D = np.sqrt(np.sum(pt_array ** 2, axis=1)[:, None] + np.sum(pt_array ** 2, axis=1)[None] - 2 * np.dot(pt_array, pt_array.T))
        D = np.nan_to_num(D, nan=0)
        return D

    def extract_collinear(self, num_coll_sets, threshold, visualize=False):
        """
        Extracts the collinear sets of the segmented fiducial markers.
        Args:
            threshold (float in range [0,1]): handles the threshold for the collinearity criteria
            visualize (bool): allows for visualization of the collinear sets
        Returns:
            coll_sets (numpy array (k,3)): contains k arrays of collinear points referenced by their label
        """
        # Initialize variables
        collinear_set_tmp = []          # Stores index of collinear points found
        num_invalid_collinear = 0       # Keeps a count of the collinear points that do not satisfy distance criteria

        # Make a copy of the cordinates of the extracted fiducial markers
        Pt = self.pt_train_coordinates.copy()   # (x,y) coordinates
        N = self.total_markers                  # Number of fiducial markers

        # Extract picewise euclidean distance between the extracted fiducial markers
        #D = self.compute_distances(Pt)
        D_sorted = np.around(np.sort(self.pt_train_distances, axis=0))   #Sort pixel fiducial distances from smallest to biggest
        D = self.pt_train_distances                    # Non sorted distances
        mD = np.mean(D_sorted[0:3, :])                 # Average distance (helps identify incongruent collinear points)

        # FIND COLLINEAR POINTS
        P = np.hstack((Pt, np.ones([N, 1])))                    # Add a column of ones (used to calculate determinant)

        # Iterate over all possible combinations
        for p1, p2, p3 in itr.combinations(range(0, N), 3):
            pt = np.vstack((P[p1], P[p2], P[p3]))         # Extract a specific combination of points

            det = np.linalg.det(pt)         # Calculate the determinant of the matrix containing the 3 points
            d1 = D[p1, p2]                   # Extracts the distance between point 1 and 2
            d2 = D[p2, p3]                   # Extracts the distance between point 2 and 3

            # Check for collinearity
            if abs(det) < (mD*self.coll_thrsh):
                # Check for distance validity
                if (d1 < mD*threshold) and (d2 < mD*threshold):
                    collinear_set_tmp.append([p1, p2, p3])
                else:
                    num_invalid_collinear += 1

        self.coll_sets = np.array(collinear_set_tmp)            # Store the collinear sets found

        if visualize:
            self.train_image.visualize_sets(self.coll_sets)               # Visualize all of the collinear sets found

        return self.coll_sets

    def Normalization(self,nd, x):
        '''
        Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).
        Input
        -----
        nd: number of dimensions, 3 here
        x: the data to be normalized (directions at different columns and points at rows)
        Output
        ------
        Tr: the transformation matrix (translation plus scaling)
        x: the transformed data
        '''

        x = np.asarray(x)
        m, s = np.mean(x, 0), np.std(x)
        if nd == 2:
            Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
        else:
            Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

        Tr = np.linalg.inv(Tr)
        x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
        x = x[0:nd, :].T

        return Tr, x

    def compute_dlt(self, train_pts):
        """
        Calculate the Direct Linear Transform (DLT) using points.

        """
        labels = self.get_labels()      # Get the labels of the segmented fiducial markers
        N = len(train_pts)              # Number of points

        # Extract the relevant points ordered according to the train_pts labels:
        real_coordinates_tmp = self.pt_original_coordinates[train_pts]
        img_coordinates_tmp = np.zeros((N, 2))
        for i, val in enumerate(train_pts):
            img_coordinates_tmp[i, :] = self.pt_train_coordinates[labels == val]

        # ADD NORMALIZATION
        real_pts = real_coordinates_tmp
        img_pts = img_coordinates_tmp
        '''
        Tr_real, real_pts = self.Normalization(3, real_coordinates_tmp)
        Tr_img, img_pts = self.Normalization(2, img_coordinates_tmp)
        '''

        A = []                          # Initialize Array
        for i in range(N):
            x, y, z = real_pts[i]
            u, v = img_pts[i]

            A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

        A = np.asarray(A)                   #Transform to numpy array
        #print(f"Shape of A: {A.shape}")

        # Find the 11 parameters:
        U, S, V = np.linalg.svd(A)

        #The smallest singular value is found at S[-1,-1], a perfect reconstruction has this value equal to 0
        print(f"Error S[-1]: {S[-1]:.2f}")

        # The parameters are in the last row of V, we can extract them and normalize
        L = V[-1, :] / V[-1, -1]

        # Camera projection matrix
        H = L.reshape(3, 4)                 # Also called P

        # DENORMALIZATION

        #H = np.dot(np.dot(np.linalg.pinv(Tr_img), H), Tr_real)
        #H = H / H[-1, -1]
        #L = H.flatten(0)

        # mean error
        uv2 = np.dot(H, np.concatenate((real_pts.T,
                                        np.ones((1, real_pts.shape[0])))))
        uv2 = uv2 / uv2[2, :]
        err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - img_pts) ** 2, 1)))

        self.DLT_params = H
        return H, err

    def evaluate_dlt(self, test_pts):
        """
        Cross validation accuracy of the computed DLT parameters performed on the test_pts.
        Args:
            test_pts (array of ints): label of the points to be used in the evaluation
        Returns:
            RMSE (float): root mean square error of the predicted distance
        """
        # Extract labels
        labels = self.get_labels()  # Get the labels of the segmented fiducial markers
        M = len(test_pts)           # Number of points used for testing

        # Extract the relevant points ordered according to the test_pts labels:
        real_coordinates = self.pt_original_coordinates[test_pts]
        img_coordinates = np.zeros((M, 2))
        for i, val in enumerate(test_pts):
            img_coordinates[i, :] = self.pt_train_coordinates[labels == val]

        #Map 3D coordinates to 2D coordinates
        tmp = np.concatenate((real_coordinates.T, np.ones((1, real_coordinates.shape[0]))))
        prediction = np.dot(self.DLT_params, tmp)
        prediction = prediction / prediction[2, :]
        prediction = prediction.T

        #Compute distancess accross points
        d_prediction = self.compute_distances(prediction[:, 0:2])
        d_real = self.compute_distances(img_coordinates[:, 0:2])

        #Compute error in distances
        diff = np.tril(d_prediction - d_real).flatten()
        diff = diff[diff != 0]

        #Compute RMSE
        Nd = len(diff)
        RMSE = np.sqrt(1/Nd * np.sum(diff**2))

        return RMSE

    def cross_eval(self,num_test_pts):
        return 0
