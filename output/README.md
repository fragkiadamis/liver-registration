# Experiments Breakdown.

| Exp | Register | Crop | Transformations     | Fixed    | Moving | Masks | Sampling | Metric | Mean   |
|-----|----------|------|---------------------|----------|--------|-------|----------|--------|--------|
| 01  | Volumes  | No   | Rigid ---> B-spline | SPECT-CT | ceMRI  | No    | Random   | MI     | 53.95% |
| 02  | Volumes  | Yes  | Rigid ---> B-spline | SPECT-CT | ceMRI  | No    | Random   | MI     | 89.84% |
| 03  | Volumes  | Yes  | Rigid ---> B-spline | SPECT-CT | ceMRI  | Yes   | Random   | MI     | 78.93% |
| 04  | Masks    | No   | Rigid ---> B-spline | SPECT-CT | ceMRI  | -     | Random   | MI     | 78.55% |