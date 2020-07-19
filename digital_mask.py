
import argparse
import cv2
import dlib
import numpy as np

# Convert dlib's shape to a numpy array
def shape_to_np(shape, dtype='int'):
    coords = np.zeros((68,2), dtype=dtype)

    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

# Find out which landmark the point belongs to
def get_index_coords(shape, pt):
    indices = np.where((shape == pt).all(axis=1))
    for i in indices[0]:
        return i
    return None

# Given an image, this function will find a random face, and extract its data
# The data is put in a form which will make it easy to reconstruct as a mask
# on another image or video
def extract_mask(frame, detector, predictor, showface=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None, None, None
    mask = np.zeros_like(gray, dtype=np.uint8)

    # we break at the end of this loop to get the first face
    # ideally there should be only one face in the image
    for rect in rects:
        # Get the landmark points, and their convex hull
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        hull = cv2.convexHull(shape)
        cv2.fillConvexPoly(mask, hull, 255)

        # Display the chosen face, if asked for
        if showface:
            cv2.imshow('frame', mask)
            cv2.waitKey(0) 
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

        # Divide the face into triangles
        rect = cv2.boundingRect(hull)
        subdiv = cv2.Subdiv2D(rect)
        for (x,y) in shape:
            subdiv.insert((x,y))
        triangles = subdiv.getTriangleList()
        tri_indices = []

        # Process the triangles, get the landmark indices for each one
        # This will make it easier to reconstruct them later
        for t in triangles:
            pt1, pt2, pt3 = ((t[0],t[1]),(t[2],t[3]),(t[4],t[5]))
            pt1_i = get_index_coords(shape, pt1)
            pt2_i = get_index_coords(shape, pt2)
            pt3_i = get_index_coords(shape, pt3)
            if None not in {pt1_i, pt2_i, pt3_i}:
                tri_indices.append([pt1_i, pt2_i, pt3_i])

        # Extract the real data
        triangles = shape[tri_indices]
        data = []
        for i in range(len(triangles)):
            (x,y,w,h) = cv2.boundingRect(triangles[i])
            mask = np.zeros((h,w), dtype=np.uint8)
            triangles[i] -= np.array([x,y])
            cv2.fillConvexPoly(mask, triangles[i], 255)
            data.append(cv2.bitwise_and(frame[y:y+h, x:x+w], frame[y:y+h, x:x+w], mask=mask))
        break

    return data, triangles, tri_indices

# quadruple to tuple of tuples
def q2t(x):
    return tuple(x[0]), tuple(x[1])

# Apply the face mask to a frame
# You need the data from extract_mask
def apply_facemask(frame, detector, predictor, facedata, face_triangles, landmark_indices, no_mouth=False, seamlessclone=False):
    #identify faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        return None

    for rect in rects:
        # initialize newframe = facemask, allfaces_mask = 1 if facemask else 0, lines = triangle lines
        newframe = np.zeros_like(frame)
        lines = set()
        exclude = []

        # get all the triangles for this face, add the hull to the mask
        shape = shape_to_np(predictor(gray, rect))
        frame_triangles = shape[landmark_indices]
        
        for i in range(len(frame_triangles)):

            # remove mouth pieces from mask, if desired
            if no_mouth and (61 in landmark_indices[i] or 62 in landmark_indices[i] or 63 in landmark_indices[i]) \
                and (65 in landmark_indices[i] or 66 in landmark_indices[i] or 67 in landmark_indices[i]):
                # make sure to remove frame_triangles[i] from allfaces_mask
                exclude.append(frame_triangles[i])
                continue

            # add edge lines to lines set
            for j in range(3):
                for k in range(j):
                    if frame_triangles[i][j][0] < frame_triangles[i][k][0]:
                        lines.add( (tuple(frame_triangles[i][j]), tuple(frame_triangles[i][k])) )
                    else:
                        lines.add( (tuple(frame_triangles[i][k]), tuple(frame_triangles[i][j])) )

            # create affine transformation from the data triangle to the frame space
            mask_triangle = face_triangles[i]
            (x,y,w,h) = cv2.boundingRect(frame_triangles[i])
            frame_triangle = (frame_triangles[i] - np.array([x,y]))
            M = cv2.getAffineTransform(mask_triangle.astype(np.float32), frame_triangle.astype(np.float32))

            # error handling for out of bounds faces
            w, h = x - max(x,0) + w, y - max(y,0) + h
            x, y = max(x,0), max(y,0), 
            framerect = newframe[y:y+h, x:x+w]
            h, w = framerect.shape[:2]
            if h == 0 or w == 0:
                continue
            
            # warp using affine transform and add to frame
            warped_triangle = cv2.warpAffine(facedata[i], M, (w,h))
            newframe[y:y+h, x:x+w] = cv2.add(framerect, warped_triangle)

        # insert mask on current rectangle
        grayframe = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
        _, face_mask = cv2.threshold(grayframe, 1, 255, cv2.THRESH_BINARY)

        if seamlessclone:
            M = cv2.moments(face_mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        
        for t in exclude:
            cv2.fillConvexPoly(face_mask, t, 0)

        # blur lines
        line_mask = np.zeros((frame.shape[0], frame.shape[1], 1))
        for line in lines:
            cv2.line(line_mask, *line, 255, 2)
        newframe_blurred = cv2.medianBlur(newframe, 5)
        newframe = np.where(line_mask != 0, newframe_blurred, newframe)

        if seamlessclone:
            try:
                # seamless clone crashes if the face mask is near the edge of the frame
                # in that situation, we don't render that particular mask
                frame = cv2.seamlessClone(newframe, frame, face_mask, (cX, cY), cv2.NORMAL_CLONE)
            except:
                pass
        else:
            invmask = ~face_mask
            frame = cv2.add(cv2.bitwise_and(newframe, newframe, mask=face_mask),\
                               cv2.bitwise_and(frame, frame, mask=invmask))

    return frame

def main(args):
    # settings
    # scale of display; larger is more computationally expensive
    rescale = args.rescale

    # should we make a hole in the mask for the mouth?
    no_mouth = args.mouth

    # should we use seamless cloning? Doesn't work well with no_mouth
    seamlessclone = args.seamlessclone

    # source image for facemask
    filename = args.face

    # should we show which face was selected for masking?
    showface = args.showface

    # where is the dlib frontal face detector stored
    predictor_name = args.predictor_name

    # Initialize dlib and opencv data structures
    # And extract the mask data
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_name)

    cap = cv2.VideoCapture(0)
    cap.read()
    cap.read()
    facemask = cv2.imread(filename)
    facedata, face_triangles, face_indices = extract_mask(facemask, detector, predictor, showface)

    if facedata is None:
        print("No faces found in source image")
        cv2.destroyAllWindows()
        return
    
    # Run video, and process the mask data
    while(1):
        ret, frame = cap.read() 

        frame = cv2.resize(frame, (int(frame.shape[1] * rescale), int(frame.shape[0]*rescale)))
        newframe = apply_facemask(frame, detector, predictor, facedata, face_triangles, face_indices, no_mouth, seamlessclone)
        if newframe is None:
            newframe = frame

        cv2.imshow('frame', newframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Put digital masks on your face')
    parser.add_argument('predictor_name', type=str, help="path to dlib's frontal face detector")
    parser.add_argument('face', type=str, help="path to the image of the face mask")
    parser.add_argument('-r', '--rescale', help="window rescale factor", type=float, default=0.5)
    parser.add_argument('-s', '--seamlessclone', help="uses the seamless clone method to merge the mask into the scene", action="store_true")
    parser.add_argument('-m', '--mouth', help="leaves a hole open in the mask for your mouth", action="store_true")
    parser.add_argument('-f', '--showface', help="show which face was identified", action="store_true")

    main(parser.parse_args())
