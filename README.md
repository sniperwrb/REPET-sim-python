# REPET-sim-python
Python translation for Rafii's Matlab code "Repet-sim"

# Update 2019-5-15
Please use "repetest.py". It is a little better.

Sim:
Limiting L2-norm of each amplitude-spectrum to <=1, and do dot product (Sim matrix = Spec * Spec.T). It works better than simply calculating correlation between two spectra (in REPET-sim), because it will not regard almost-silent frames as similar with normal frames.

We use shorter frame lengths to calculate the dot product, and then do a windowed summation (convolution) along the main diagonal. It is the same as if the frame length is very long, but it saves a lot of memory and some running time.

Mask:
We only use a few (maybe dozens) most similar frames which are far (at least dozens of frames) away from each other, so it is much faster than original REPET-sim which may have thousands of similar frames per frame using the default hyperparameters.
