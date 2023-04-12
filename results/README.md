# Experiments Breakdown.

| Exp | Register | Crop | Transformations     | Fixed    | Moving | Masks | Sampling | Metric | Mean   |
|-----|----------|------|---------------------|----------|--------|-------|----------|--------|--------|
| 01  | Volumes  | No   | Rigid ---> B-spline | SPECT-CT | ceMRI  | No    | Random   | MI     | 53.95% |
| 02  | Volumes  | Yes  | Rigid ---> B-spline | SPECT-CT | ceMRI  | No    | Random   | MI     | 89.84% |
| 03  | Volumes  | Yes  | Rigid ---> B-spline | SPECT-CT | ceMRI  | Yes   | Random   | MI     | 78.93% |
| 04  | Masks    | No   | Rigid ---> B-spline | SPECT-CT | ceMRI  | -     | Random   | MI     | 78.55% |

For experiment 02, I first changed the spacing of the voxels to each axis with the smallest between the respective CT 
and MRI. After, each study was cropped based on the boundaries of the liver mask. Intensity based registration was 
applied on the cropped volumes. The pipeline was Rigid MI and Bspline MI.