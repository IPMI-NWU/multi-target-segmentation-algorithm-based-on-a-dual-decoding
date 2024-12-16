# Multi-target segmentation algorithm based on a dual decoding

To address the difficulty of multi-target segmentation tasks with partial annotations, a multi-target segmentation algorithm based on a dual-decoder structure is proposed. The algorithm uses a single encoder with a dual-decoder structure as the basic framework, which can integrate multiple partially annotated datasets for network training. 

By considering the overlap size and feature similarity between the segmented targets and lung fields, the segmentation targets are divided into two sets. The dual-decoder structure then separates the feature information extracted by the encoder, reducing the impact of excessive overlap in the target areas on segmentation performance.

# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* Python == 3.10.14
* numpy == 1.26.4