# pytorch morphological dilation2d and erosion2d
An implementation for image (tensor) morphological dilation2d and erosion2d in pytorch

We implement the morpholgical dilation2d erosion2d for 4D input tensor. The implementation uses convolution-like operator torch.nn.unfold().

Ref:
[1] A tensorflow implementation. "Morphological Networks for Image De-raining (Ranjan Mondal et al. (2019))" https://github.com/ranjanZ/2D-Morphological-Network   
[2] A vector (image is flatten to vector) pytorch implementation "Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019))" https://github.com/jlebensold/iclr_2019_buffalo-3 
