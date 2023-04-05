# Experiments Breakdown.

|              | Crop | Transformations                            | Fixed    | Moving | Masks | Sampling | Metric |
|--------------|------|--------------------------------------------|----------|--------|-------|----------|--------|
| Experiment 1 | Yes  | Rigid ---> B-spline                        | SPECT-CT | ceMRI  | No    | Random   | MI     |
| Experiment 2 | Yes  | Rigid ---> B-spline                        | SPECT-CT | ceMRI  | Yes   | Random   | MI     |
| Experiment 3 | No   | Rigid ---> B-spline                        | SPECT-CT | ceMRI  | No    | Random   | MI     |

Experiment 0 is the liver masks registered and then the transform is applied on the image volume.

Experiment 3, minimum spacing for each axis.

The masks bring the dice down...

The failures still persist in the full images when I use the masks.