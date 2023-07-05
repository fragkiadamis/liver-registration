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
registrations beforehand. The objective of this application is to:
- A first registration between the baseline ceMRI and SPECT/CT images (pre-treatment dosimetry),
- A second registration between the baseline ceMRI and the PET/CT images (post-treatment dosimetry). 

## Database Creation
The database of this application is consisted by files in DICOM format. In order to be able to perform the necessary 
image processing techniques, we need to convert the files into NIfTI format.

### Structure
Each patient in the dataset has a ceMRI, a SPECT-CT and a PET-CT scan. For each modality we have available the 
volume sequence, the RT structures (liver & tumor segmentations) as well as a manual registration performed by a 
specialist. The application is developed based on a specific structure that the input database (DICOM files) 
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
execute the file by using the command ```python database.py -i data/dicom -o data/nifti```. The arguments ```-i``` and ```-o``` 
are for the input path and the output directory path, respectively. After the extraction procedure finishes, the 
nifty data will be available at the defined directory. The extracted data have a similar structure to the DICOM.
In this example, the data will have the following structure.

```commandline
|── data
│   ├── dicom
│   ├── nifti
│   │   ├── patient_001
│   │   │   ├── ct_liver.nii.gz
│   │   │   ├── ct_tumor.nii.gz
│   │   │   ├── ct_volume.nii.gz
│   │   │   ├── mri_liver.nii.gz
│   │   │   ├── mri_tumor.nii.gz
│   │   │   ├── mri_volume.nii.gz
│   │   ├── ...
│   │   ├── patient_XXX
│   │   │   ├── ct_liver.nii.gz
│   │   │   ├── ct_tumor.nii.gz
│   │   │   ├── ct_volume.nii.gz
│   │   │   ├── mri_liver.nii.gz
│   │   │   ├── mri_tumor.nii.gz
│   │   │   ├── mri_volume.nii.gz
```
The volume is the acquired sequence for each modality and tumor & liver are the segmentations of the respective 
anatomies. Some patients may have more data available. For example the tumor segmentations might be in multiple files.
The database.py has a function `add_segmentations()` in order to add the segmentations in one file. In some cases
the SPECT-CT might have multiple segmentations of the liver (delineation on the CT and delineation on the MRI which is 
manually registered on the CT). In this case the nii files are named as ct_liver_1.nii.gz, ct_liver_2.nii.gz etc.
By looking at the logs someone can see the respective DICOM RTStruct file for each nii.gz file and decide which are
going to be deleted and which one will be kept. The kept one is better to be renamed manually to ct_liver.nii.gz for
consistency and to avoid possible problems with the rest of the scripts.

## Preprocessing
In the preprocessing.py file several functions are defined that are necessary to be performed before the registration.
The execution is done by this command `python preprocessing.py -i data/nii -o data/nii_preprocessed -t XXX`. The `-i` is the input
directory and the `-o` respectively the output directory. 

The `-t` is to define the "reason" of the preprocessing. If for example someone wants to preprocess the nii files for
registration with elastix then it should be `-t elx`.

For the segmentation of the ct images using Felix's unet3d architecture then it should be `-t dls` and finally,

for deep learning registration using the LocalNet or the GlobalNet, then it should be `-t dlr`. In the case that the data
are preprocessed for use by the LocalNet, then the moving images should come from the pre-aligned MRIs (e.g. from elastix)
so the user must also define the pre-aligned directory `-pd data/elastix/baseline_unet3d_masks`.

For each preprocessing operation the user can decide which functions to use by commenting uncommenting the evocation.

## Postprocessing
The post-processing file is similar to the preprocessing file. The execution command is `python postprocessing.py -i data/localnet -o data/nii_iso -t XXX`.
It is used for the data post-processing after the deep learning training sessions. Use `-t reg` for postprocessing
from the registration architectures and `-t seg` for the segmentation architectures.

## Elastix Registration
To register a CT-MRI pair of a patient use the registration.py file. To use this file we need to define the registration
pipeline in a json file in the pipelines directory. In this json all the details of the registration are included.
1. fixed and moving studies.
2. step
   1. image names for fixed and moving modalities.
   2. The parameter's file name (Obviously there is a parameters directory in the root directory to save the elastix parameter files).
   3. If the deformation field is going to be extracted or not
   4. and to which images to apply the registration after.
3. Define the base filenames of the ground truth images for evaluation.
Check the included files to get the idea...

The execution of the registration.py is done by the command `python registration.py -i data/nii_preprocessed -o data/elastix -pl pipelines/pipeline.json -p JohnDoe_ANONXXXXX`.
`-i` is the input directory, `-o` is the output directory, `-pl` is the pipeline that is going to be used and finally `-p` the patient that is going to be registered. 
In this file the CT-MRI pair is selected for a specific patient and then:
1. Calculates the initial metrics (before registration)
2. Perform the registration according to the defined pipeline
3. Calculate the pipeline's step metrics
4. Move on the next step of the pipeline and so on...
5. Save the metrics for each step in an evaluation.json file.

## Excel File
To create an excel file containing the aggreagated results of the elastix registration then execute the excel_file.py script.
To execute the script use the command `python excel_file -i data/elastix -pl baseline_unet3d_ct_seg` where `-i` is the input directory
(contains the patients after registration with the evaluation.json files), `-o` is the output of the excel file (where the file is going to be stored and its name)
and finally `-pl` is the name of the pipeline that is going to be evaluated.

You can modify this file accordingly to aggreagate the results into an excel for the deep learning segmentation and deep learning registration processes.

## Plots
Use the plot_statistics.py file to create plots from the excel files. Execution: `python plot_statistics.py -i results/registration_results.xlsx -o plots`
where `-i` is the input excel file containing the data nad `-o` is the directory where the plots are going to be saved. Modify accordingly if needed...

## Deep Learning Registration
The localnet.py file is containing the necessary architecture of the LocalNet from monai framework. Run this file in order to
to start training the network. Execution: `python localnet.py -i data/localnet -o data/models -f 0 -n localnet`. `-i`
is the input directory (contains the pre-processed data for deep learning registration use), `-o` is the output directory
(where the model is going to be saved), `-f` define the fold that is going to be trained in this session, `-n` the name of the architecture.

What applies to localnet.py also applies for globalnet.py e.g. `python globalnet.py -i data/globalnet -o data/models -f 0 -n globalnet`