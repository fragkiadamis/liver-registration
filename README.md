# Multimodal Liver Registration --- Intra-patient CT & ceMRI.

## Introduction
This application is part of the Deep lEarning tooLs for selectIve iNtErnal rAdiation ThErapy of hepatic tumours 
(DELINEATE) project. hepatocellular carcinoma (HCC) can be treated by selective internal radiation therapy (SIRT), which
consists in injecting selectively into the hepatic arteries yttrium-90 (<sup>90</sup>Y) β-radiation emitter 
microspheres. Prior to 90Y bead injection, several examinations must be performed. First, a baseline contrast­enhanced
magnetic resonance imaging (ceMRI) scan is acquired to visualise the tumour, and a hepatic angiography is acquired to
identify the extrahepatic vessels that must be prophylactically embolized to preserve healthy organs. Then, a simulation
of the treatment is performed by injecting Technetium-99m macroaggregated albumin (<sup>99m</sup>Tc-MAA) as a surrogate 
for 90Y particles. A pre-treatment dosimetry is performed by acquiring a single-photon emission computed tomography 
(SPECT/CT) scan which allows to obtain the 99mTc-MAA distribution of activity and to calculate the 90Y activity to 
prescribe. Once calculated, the appropriate amount of 90Y microspheres is injected into the patient and a positron 
emission tomography (PET/CT) scan is acquired to ensure the proper <sup>90</sup>Y distribution and calculate 
<sup>90</sup>Y post-treatment dosimetry. To be able to calculate proper dosimetry, it is necessary to perform 
registrations beforehand.

## Database Creation
The database of this application is consisted by files in DICOM format. In order to be able to perform the necessary 
image processing techniques, we need to convert the files into NIfTI format.

### Structure
Each patient in the dataset has a ceMRI, a SPECT-CT and a PET-CT scan. For each modality we have available the 
volume sequence, the RT structures (liver & tumor segmentations) as well as a manual registration performed by a 
specialist. The application is developed based on the a specific structure that the input database (DICOM files) 
must follow. The dicom database is in a directory named dicom inside the data directory. The data directory should 
be at the root directory. The structure is the following.

```commandline
├── data
│   ├── dicom
│   │   ├── patient_001
│   │   │   ├── ceMRI
│   │   │   │   ├── DICOM series
│   │   │   │   ├── Registration
│   │   │   │   ├── RT Structure
│   │   │   ├── SPECT-CT
│   │   │   │   ├── DICOM series
│   │   │   │   ├── Registration
│   │   │   │   ├── RT Structure
│   │   │   ├── PET-CT
│   │   │   │   ├── DICOM series
│   │   │   │   ├── Registration
│   │   │   │   ├── RT Structure
│   │   ├── ...
│   │   ├── patient_XXX
│   │   │   ├── ceMRI
│   │   │   │   ├── ...
│   │   │   ├── SPECT-CT
│   │   │   │   ├── ...
│   │   │   ├── PET-CT
│   │   │   │   ├── ...
```
The database.py file is responsible for extracting the database from teh DICOM files into a NIfTI format. You can 
execute the file by using the command ```python -i data/dicom -o data/nifti```. The arguments ```-i``` and ```-o``` 
are for the input path and the output directory path, respectively. After the extraction procedure finishes, the 
nifty data will be available at the defined directory. The extracted data have a similar structure like the DICOM 
data but one layer less. In this example, the data will have the following structure.

```commandline
├── data
│   ├── dicom
│   ├── nifti
│   │   ├── patient_001
│   │   │   ├── ceMRI
│   │   │   │   ├── liver.nii.gz
│   │   │   │   ├── tumor.nii.gz
│   │   │   │   ├── volume.nii.gz
│   │   │   ├── SPECT-CT
│   │   │   │   ├── liver.nii.gz
│   │   │   │   ├── tumor.nii.gz
│   │   │   │   ├── volume.nii.gz
│   │   │   ├── PET-CT
│   │   │   │   ├── liver.nii.gz
│   │   │   │   ├── tumor.nii.gz
│   │   │   │   ├── volume.nii.gz
│   │   ├── ...
│   │   ├── patient_XXX
│   │   │   ├── ceMRI
│   │   │   │   ├── ...
│   │   │   ├── SPECT-CT
│   │   │   │   ├── ...
│   │   │   ├── PET-CT
│   │   │   │   ├── ...
```
The volume is the acquired sequence for each modality and tumor & liver are the segmentation of the respective 
anatomies. Some patients may have more data available, like more than one liver or tumor segmentations or a 
segmentation of the necrotic tissue. Either, the application uses only the files with the names that are 
specified in the above structure. Of course with the correct file renaming or the refactoring of the code, each user 
can utilise different data.

