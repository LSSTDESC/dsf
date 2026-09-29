Hankel Transforms
=================

DSF uses Hankel transforms to compute various quantities throughout the
pipeline. The code provides a module, ``dsf.hankel``, with a unified 
public interface for performing these transforms. The public layer is
accessed through the ``dsf.hankel.hankel.HankelTransform`` class.

There are three backend implementations available:

1. ``fftlog``: Uses the FFTLog algorithm to compute the transform.
It supports projected and spherical 1D Hankel transforms. This is 
the fastest option for most 1D applications.

2. ``matrix_zeros``: Uses identities of Bessel zeros to compute 
Hankel transform matrix operators. It supports projected 1D, 2D, and 3D 
Hankel transforms. Can exhibit ringing at small scales.

3. ``matrix_direct``: Uses a user-defined k- and r-grid to compute
Hankel transform matrix operators. It supports projected 1D and 2D 
Hankel transforms. Requires fine-tuning of the k- and r-sampling upon 
generation. Assuming the grid is sampled well, the method is robust
against ringing at small scales, but can exhibit minor ringing at large 
scales.

As a rule of thumb, use ``fftlog`` to perform any 1D Hankel transforms.
For any 2D transforms, try ``matrix_zeros`` if you don't need to sample at 
particularly small scales; otherwise try ``matrix_direct``. However, we 
recommend trying both backends to explicitly check for ringing behavior 
and speed.

*Common Issue:* When performing a matrix Hankel transform, the the input 
k-grid for the power spectrum **must** cover the full range of the k-values
used to generate the Hankel transform matrix. The ``matrix_zeros`` backend 
can dynamically expend its k-grid during generation, depending on the other 
initialization parameters, meaning that the required k-grid for the power 
spectrum may be larger than you expect!