### Explorations with Cam/Computer Vision ###
This repo is for explorations with python cam/computer vision packages. 
2 projects are included:
- **CamHero** - a simple app that tracks movement with a webcam and user tries to touch squares when cirles appear in them, triggering a sound; very, very rudimentary approach presently
- **Pytheramin** - a simple attempt to somewhat mimic the behavior of a theramin using a webcam and a few python packages

### CamHero ###
Start by running `python camhero.py` from the command line.
The game is simple: when a circle appears in a square, touch the square to trigger a sound.
The game is over when the user misses 5 circles.
Press `q` to quit

### Pytheramin ###
Start by running `python pytheramin.py` from the command line.
Good lighting is important for this app.
When starting, make sure hand is prominent in the frame.
The object that is being tracked is indicated by a green square
The pitch of the sound is determined by the vertical position of the tracked object
The vibrato of the sound is determined by the horizontal position of the tracked object
Backlighting may cause the object to be tracked to be lost or to lock onto elements of the background
Current sound is not very pleasant, plan to replace with a better sound and/or add a sound selection option
Press `q` to quit
