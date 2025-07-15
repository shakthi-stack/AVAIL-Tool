Step 0: Env set up and running 
conda activate face-detection-tool
Step 1:
Automatically detect faces in the video you provide

python automatic_detection.py workspace/example_video.mp4 workspace/example_video.pickle

Multiple face detection models > -d [s3fd, haar, mtcnn, dfl]

For the current case we just use haar (which is less accurate but very fast compared to the others)

Run:

python automatic_detection.py workspace/example_video.mp4 workspace/example_video.pickle -d haar


Step 2:
Split up the videos into frames to run the manual annotation tool. 

Run: 

python create_frames_directory.py workspace/example_video.mp4 workspace/frames/

Step 3: (Collaborative Session Focus)
Manual Annotation GUI

python manual_annotation.py workspace/example_video.mp4 workspace/example_video.pickle workspace/frames

Two phases:

Step 3.1:
Mark the ground truth regions > mark the start and end frames of when the face appears on the screen which will later be used further down the pipeline to guide the program when to attempt interpolation on the faces.

S to start, E to end (bottom green bar indicates the start and not started set of frames)
Navigation through the frames "<" and ">" for singles "," and "." for skip 10. 
Print Statistics by pressing the spacebar
Save the selection by pressing Enter
Close the GUI tool to open the next screen

Step 3.2:
Annotate face chains > user can label the key subject of the video and eliminate false positives and manually add new faces to assist the automatic interpolation pass later. 

Face chains (bounding boxes) are numbers 0 onwards. Press the number to toggle off or on the selected region (false positives)

Same navigations as the previous step. Users can use "k" and "l" o navigate to keyframes they marked in the previous step. Bar with color dark green indicated if the current frame is part of the keyframe.

Non key subjects detected can be deactivates by toggling the tag or the user can add new bounding boxes when they are not detected. 

Create new faces annotations by dragging the mouse to create a box and press tab.


Work is saved by pressing enter every time you make a change. 

GUI closes once you exit the screen. 


Step 4: 
Final Interpolation runs 

Step 5:
Generate the blurred output
python render_result.py workspace/example_video.mp4 workspace/blurred_video.mp4 workspace/example_video.pickle -b subject_face

-b [none, subject_face, all_faces, whole_frame]