# Data Analysis

## DICOM Patient List:
|                   |                   |                   |                   |                   |                   | 
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| JohnDoe_ANON80520 | JaneDoe_ANON82950 | JohnDoe_ANON42529 | JohnDoe_ANON27373 | JohnDoe_ANON39080 | JohnDoe_ANON51834 |
| JohnDoe_ANON65060 | JohnDoe_ANON27183 | JohnDoe_ANON15323 | JaneDoe_ANON25911 | JohnDoe_ANON77471 | JaneDoe_ANON34438 | 
| JohnDoe_ANON35169 | JohnDoe_ANON59591 | JohnDoe_ANON86311 | JohnDoe_ANON77296 | JohnDoe_ANON55831 | JohnDoe_ANON57371 | 
| JohnDoe_ANON55215 | JohnDoe_ANON39011 | JohnDoe_ANON11762 | JaneDoe_ANON83544 | JohnDoe_ANON81710 | JohnDoe_ANON23808 |
| JohnDoe_ANON84994 | JohnDoe_ANON28177 | JohnDoe_ANON45396 | JohnDoe_ANON91519 | JohnDoe_ANON61677 | JohnDoe_ANON98854 |
| JohnDoe_ANON62642 | JohnDoe_ANON87212 | JohnDoe_ANON92634 | JohnDoe_ANON23001 | JohnDoe_ANON21673 | JohnDoe_ANON72295 |
| JohnDoe_ANON83160 | JohnDoe_ANON24065 | JohnDoe_ANON53833 | JohnDoe_ANON74328 | JohnDoe_ANON15860 | JohnDoe_ANON50337 |
| JohnDoe_ANON99601 | JohnDoe_ANON96978 | JohnDoe_ANON78721 | JohnDoe_ANON55240 | JohnDoe_ANON64482 | JaneDoe_ANON47965 |
| JohnDoe_ANON29513 | JohnDoe_ANON44625 | JaneDoe_ANON12304 | JohnDoe_ANON87883 | JohnDoe_ANON70417 | JaneDoe_ANON56995 |
| JohnDoe_ANON76802 | JohnDoe_ANON36736 | JohnDoe_ANON87639 | JohnDoe_ANON87928 | JohnDoe_ANON45696 | JohnDoe_ANON55098 |
| JohnDoe_ANON10507 | JohnDoe_ANON32161 | JaneDoe_ANON56370 | JohnDoe_ANON13231 | JohnDoe_ANON22228 |                   |
| JaneDoe_ANON69091 | JohnDoe_ANON92476 | JohnDoe_ANON98767 | JohnDoe_ANON27417 | JohnDoe_ANON46160 |                   |

## Patient with problematic labels (keep out of the dataset):
1. JohnDoe_ANON77471

## Patients that the tumor is visible on the CT Scan (not all are confirmed):
1. JohnDoe_ANON45696               
2. JohnDoe_ANON80520               
3. JohnDoe_ANON27373      
4. JohnDoe_ANON22228
5. JohnDoe_ANON55215
6. JohnDoe_ANON46160
7. JohnDoe_ANON23001
8. JohnDoe_ANON86311
9. JohnDoe_ANON29513
10. JohnDoe_ANON98854
11. JohnDoe_ANON45396
12. JaneDoe_ANON56370
13. JaneDoe_ANON56995
14. JaneDoe_ANON11762
15. JohnDoe_ANON65060

## GlobalNet Validation Folds.
|      Fold 0       | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 
|:-----------------:|:------:|:------:|:------:|:------:|
| JohnDoe_ANON98854 |        |        |        |        |
| JohnDoe_ANON80520 |        |        |        |        | 
| JohnDoe_ANON65060 |        |        |        |        | 
| JohnDoe_ANON62642 |        |        |        |        |
| JohnDoe_ANON29513 |        |        |        |        |
| JohnDoe_ANON87212 |        |        |        |        |
| JohnDoe_ANON32161 |        |        |        |        |
| JohnDoe_ANON15323 |        |        |        |        |
| JaneDoe_ANON83544 |        |        |        |        |
| JohnDoe_ANON87883 |        |        |        |        |
| JohnDoe_ANON87928 |        |        |        |        |
| JaneDoe_ANON69091 |        |        |        |        |
| JohnDoe_ANON46160 |        |        |        |        |
| JohnDoe_ANON23808 |        |        |        |        |

## LocalNet Validation Folds.
|      Fold 0       |      Fold 1       |      Fold 2       |      Fold 3       |      Fold 4       | 
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| JohnDoe_ANON98854 | JohnDoe_ANON55215 | JohnDoe_ANON70417 | JohnDoe_ANON50337 | JohnDoe_ANON83160 |
| JohnDoe_ANON80520 | JohnDoe_ANON76802 | JohnDoe_ANON35169 | JohnDoe_ANON55098 | JohnDoe_ANON10507 | 
| JohnDoe_ANON65060 | JohnDoe_ANON59591 | JohnDoe_ANON84994 | JohnDoe_ANON99601 | JohnDoe_ANON27183 | 
| JohnDoe_ANON62642 | JohnDoe_ANON11762 | JohnDoe_ANON24065 | JohnDoe_ANON28177 | JohnDoe_ANON39011 |
| JohnDoe_ANON29513 | JohnDoe_ANON45396 | JaneDoe_ANON82950 | JohnDoe_ANON86311 | JohnDoe_ANON96978 |
| JohnDoe_ANON87212 | JohnDoe_ANON53833 | JohnDoe_ANON36736 | JohnDoe_ANON92634 | JohnDoe_ANON44625 |
| JohnDoe_ANON32161 | JohnDoe_ANON87639 | JohnDoe_ANON92476 | JaneDoe_ANON56370 | JaneDoe_ANON12304 |
| JohnDoe_ANON15323 | JohnDoe_ANON91519 | JohnDoe_ANON42529 | JaneDoe_ANON25911 | JohnDoe_ANON27373 |
| JaneDoe_ANON83544 | JohnDoe_ANON55240 | JohnDoe_ANON78721 | JohnDoe_ANON23001 | JohnDoe_ANON55831 |
| JohnDoe_ANON87883 | JohnDoe_ANON13231 | JohnDoe_ANON77296 | JohnDoe_ANON39080 | JohnDoe_ANON81710 |
| JohnDoe_ANON87928 | JohnDoe_ANON27417 | JohnDoe_ANON74328 | JohnDoe_ANON61677 | JohnDoe_ANON64482 |
| JaneDoe_ANON69091 | JohnDoe_ANON21673 | JohnDoe_ANON15860 | JohnDoe_ANON51834 | JohnDoe_ANON45696 |
| JohnDoe_ANON46160 | JohnDoe_ANON98767 | JohnDoe_ANON72295 | JohnDoe_ANON57371 | JohnDoe_ANON22228 |
| JohnDoe_ANON23808 | JaneDoe_ANON34438 | JaneDoe_ANON56995 |         -         |         -         |
