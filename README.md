# digital-mask
A pet project for imposing masks digitally on a video

With this project, you can take an image of a person or object that resembles a face and superimpose it onto your own face as a mask, in a real-time video.

## Requirements:
* Python 3
* OpenCV
* Dlib
* NumPy

You will need to download dlib's shape predictor from http://dlib.net. This can be done with the command:
```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

## Run the code
Prepare an image of the person / face-like object that you would like to wear as a digital mask. A large image with a prominent face will yeild more realistic results. 
Then, run on the command line
```
python digital_mask.py <path_to_dlib_shape_predictor> <path_to_face_mask_image> [-r RESCALE] [-smf]
```
Options:
* The `-r` flag allows you to change the rescale factor of the window. A larger rescale value will result in a larger window, but it will also take more compute resources, so it can be more laggy.
* The `-s` flag allows you to use the seamless cloning algorithm. This changes the way in which the face mask is pasted onto the frame. It can make pasting a face look more realistic.
* The `-m` flag will cut a hole in the face mask so that the user's mouth is displayed. It makes the movements of the mouth more realistic. 
* The `-f` flag can be run if there are multiple faces in the image and the user wants to know which face was identified by the program. Before running the video, it will display two images. The first is a black and white mask of the region of the identified face, and the second is the image of the face. You can continue the program by pressing any key.
<br/>
Warning: The `-m` and `-s` flags do not pair well together, and running them at the same time may yeild undesirable results.
