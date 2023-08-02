# Generate random EG3D faces and corresponding sketches.
python gen_sketch_sample.py --dir ./results/edit/ --seed 30840 --angle_y -0.36 --angle_p -0.36

# Generate initial 3D faces latents
python inference_initial_face.py --dir ./results/edit/ --img appear.png --sketch sket.png --mask mask.jpg

# Sketch, image and voxel optimization
python edit_optimize.py --dir ./results/edit/ --img appear.png --sketch sket.png --mask mask.jpg

