# SeggyNet

A segmentation model based on Resnet18's Multi Human Segmentation Model (resnet18 mhp 640x360). It's purpose is to demonstrate the potential of segmentation ai in the field of real-time computer graphics.

![tanav colourful](https://github.com/sagmeister101/SeggyNet/assets/173954198/887198bc-4594-4670-9beb-fbf1abeae878)



## The Algorithm

SegNet is a deep neural network for semantic segmentation, employing an encoder-decoder architecture. It reduces image dimensions in the encoder, uses pooling indices for precise upsampling in the decoder, and classifies pixels into predefined classes with softmax activation. 

![image](https://github.com/sagmeister101/SeggyNet/assets/173954198/ff99ff39-ce06-47a4-ba5c-84d5caad761d)
![image](https://github.com/sagmeister101/SeggyNet/assets/173954198/59571480-c2a2-442a-869f-33f1059f588a)


This model in particular is trained to recognize and distinguish human components. For instance, it can recognize a face, a shirt, or any of the classes of object listed below.

![image](https://github.com/sagmeister101/SeggyNet/assets/173954198/d417be69-a61d-4f43-83e1-45e341323b10)

![hand](https://github.com/sagmeister101/SeggyNet/assets/173954198/c3034334-0fb0-4bd8-a603-fedfe0df65d3)

![group](https://github.com/sagmeister101/SeggyNet/assets/173954198/0808c005-0203-40b9-a94d-c96d98319ed4)


## Running this project

Step 1.Make sure you have the jetson-inference library installed. If not, do so first.

  https://github.com/dusty-nv/jetson-inference/blob/master/README.md

Step 2. Navigate to the jetson-inference directory and run the docker.

![image](https://github.com/sagmeister101/SeggyNet/assets/173954198/a5d3db28-3611-4fa6-9b80-4980378d521e)


Step 3. download seggynet.py and segnet_utils.py from the git/main branch.(It doesn't matter which directory you put these in)


Step 4. Run the program using the follow command line ([YOUR INPUT] is whatever your camera device is named):
   python3 seggynet.py [YOUR INPUT] --visualize overlay,mask --network=fcn-resnet18-mhp-640x360


##Video Demonstration

  https://www.youtube.com/watch?v=MBtivYk0QFk
    

