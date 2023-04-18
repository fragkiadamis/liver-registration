# Pipeline & Experiments.

| Pipeline  |  Fixed   | Moving | Isotropic <br> Spacing |               Global                | Global <br> Masks |           Local            |                     Local <br> Masks                     | Final Mean |
|:---------:|:--------:|:------:|:----------------------:|:-----------------------------------:|:-----------------:|:--------------------------:|:--------------------------------------------------------:|:----------:|
| exp1.json | SPECT-CT | ceMRI  |        &cross;         |       Rigid MI <br> (Volumes)       |      &cross;      | B-spline MI <br> (Volumes) |                         &cross;                          |   63.68%   |
| exp2.json | SPECT-CT | ceMRI  |        &cross;         |   Rigid KS <br> (Bounding boxes)    |      &cross;      | B-spline MI <br> (Volumes) | <ul><li>Fixed liver_bb</li><li>Moving liver_bb</li></ul> |   87.67%   |
| exp3.json | SPECT-CT | ceMRI  |        &cross;         |        Rigid KS <br> (Masks)        |      &cross;      | B-spline MI <br> (Volumes) | <ul><li>Fixed liver_bb</li><li>Moving liver_bb</li></ul> |   89.35%   |
| exp4.json | SPECT-CT | ceMRI  |        &cross;         | Rigid KS <br> (Mask - Bounding Box) |      &cross;      | B-spline MI <br> (Volumes) |    <ul><li>Fixed liver</li><li>Moving liver</li></ul>    |   83.54%   |
| exp5.json | SPECT-CT | ceMRI  |        &cross;         |        Rigid KS <br> (Masks)        |      &cross;      |  B-spline KS <br> (Masks)  |                         &cross;                          |   95.81%   |
| exp6.json | SPECT-CT | ceMRI  |        &check;         |        Rigid KS <br> (Masks)        |      &cross;      |  B-spline KS <br> (Masks)  | <ul><li>Fixed liver_bb</li><li>Moving liver_bb</li></ul> |   96.41%   |
| exp7.json | SPECT-CT | ceMRI  |        &check;         |        Rigid KS <br> (Masks)        |      &cross;      | B-spline KS <br> (Volumes) | <ul><li>Fixed liver_bb</li><li>Moving liver_bb</li></ul> |   89.60%   |
