# SketchFaceNeRF: Sketch-based Facial Generation and Editing in Neural Radiance Field<br><sub>Official implementation</sub>

![Teaser image](./img/teaser.jpg)

## Abstract
Realistic 3D facial generation based on Neural Radiance Fields (NeRFs) from 2D sketches benefits various applications.
	Despite the high realism of free-view rendering results of NeRFs, it is tedious and difficult for artists to achieve detailed 3D control and manipulation.
	Meanwhile, due to its conciseness and expressiveness, sketching has been widely used for 2D facial image generation and editing.
	Applying sketching to NeRFs is challenging due to the inherent uncertainty for 3D generation with 2D constraints, a significant gap in content richness when generating faces from sparse sketches, and potential inconsistencies for sequential multi-view editing given only 2D sketch inputs.
	To address these challenges, we present SketchFaceNeRF, a novel sketch-based 3D facial NeRF generation and editing method, to produce free-view photo-realistic images. 
	To solve the challenge of sketch sparsity, we introduce a Sketch Tri-plane Prediction net to first inject the appearance into sketches, thus generating features given reference images to allow color and texture control. 
	Such features are then lifted into compact 3D tri-planes to supplement the absent 3D information, which is important for improving robustness and faithfulness.
	However, during editing, consistency for unseen or unedited 3D regions is difficult to maintain due to limited spatial hints in sketches.
	We thus adopt a Mask Fusion module to transform free-view 2D masks (inferred from sketch editing operations) into the tri-plane space as 3D masks, which guide the fusion of the original and sketch-based generated faces to synthesize edited faces. 
	We further design an optimization approach with a novel space loss to improve identity retention and editing faithfulness. 
	Our pipeline enables users to flexibly manipulate faces from different viewpoints in 3D space, easily designing desirable facial models. 
	Extensive experiments validate that our approach is superior to the state-of-the-art 2D sketch-based image generation and editing approaches in realism and faithfulness. 

## Code

Coming soon
