#brain extraction and register it to MNI152 template
#as one of the steps in pre-processing

rm(list=ls())
library(scales)
library(neurobase)
library(fslr)
library("extrantsr")

setwd("~/Projects/MRI_Autoencoder")
MNI_templatePath <-"~/Data/ADNI/eva/"

removeSubset <- function(primary, subset){
  new_vec = c()
  for (v2 in primary){
    v2_split <- unlist(strsplit(v2, ".nii"))[[1]]
    v2_split <- unlist(strsplit(v2_split, "/"))
    v2_split <- v2_split[length(v2_split)]
    append_val = TRUE
    for (v1 in subset){
      v1_split <- unlist(strsplit(v1, "_MNI"))[[1]]
      if (v2_split == v1_split){
        append_val = FALSE
      }
    }
    if (append_val){
      new_vec <- c(new_vec, v2)
    }
  }
  return(new_vec)
}

normalise_MIN152 <- function(inT1FilePath, MNI_t1_brain, outDir){
  
  print(paste0("current input file: ", inT1FilePath, " ***"))
  print(inT1FilePath)
  #read t1 path
  t1 <- readnii(inT1FilePath,reorient = FALSE)
  # brain extraction
  bet1_T1 = fslbet(t1,retimg = TRUE)
  double_ortho(t1,bet1_T1)
  # b: Find the gravity center of bet_T1_v1
  cog <- cog(bet1_T1,ceil = TRUE)
  cog <- paste("-c",paste(cog,collapse = " "))
  #c: re-run at center new cog
  bet2_T1 = fslbet(bet1_T1,retimg = TRUE,opts = cog)
  double_ortho(t1,bet2_T1)
  
  # register t1 to MNI template T1 space
  t1 <-nii.stub(inT1FilePath)
  print(paste0("main file name: ", t1))
  tks<- strsplit(inT1FilePath,"/")[[1]]
  t1MainName <- nii.stub(tks[length(tks)])
  t12MNI152FileName <-paste0(t1MainName,"_MNI.nii.gz")
  t12MNI152FilePath <- file.path(outDir,t12MNI152FileName)
  
  MNI_regT12TemplateT1<- ants_regwrite(filename = bet2_T1,template.file = MNI_t1_brain,
                                       outfile = t12MNI152FilePath,
                                       verbose = FALSE,
                                       typeofTransform = "SyN"
  )
  double_ortho(MNI_regT12TemplateT1,MNI_t1_brain)
  return(MNI_regT12TemplateT1)
}#End function


#step 1: loading MNI152 template
MNI_T1_1mm_brain_File <- file.path(MNI_templatePath,"MNI152_T1_1mm_brain_181x217x181.nii")
MNI_t1_brain <- readnii(MNI_T1_1mm_brain_File,reorient = FALSE)
orthographic(MNI_t1_brain)

#step 2: check output directory
normPath <-"~/Data/ADNI/2Yr_1.5T_norm"
if(dir.exists(normPath)){
  #unlink(regPath,recursive = TRUE)
  print(paste(normPath, " directory exists."))
} else {
  dir.create(normPath)
}

#step 3: load input file
inFileDir <- "ADNI1_Annual_2_Yr_1.5T/ADNI/137_S_1414/MPR__GradWarp__N3__Scaled/2009-08-26_11_06_33.0/S72806"
inFilePath <- file.path(inFileDir,"ADNI_137_S_1414_MR_MPR__GradWarp__N3__Scaled_Br_20100113101118651_S72806_I163393.nii")
tks<- strsplit(inFilePath,"/")[[1]]
inFileMainName <- nii.stub(tks[length(tks)])
t1_MNI152 <-paste0(inFileMainName,"_MNI.nii.gz")
# invoking the function normalise_MIN152
normalise_MIN152(inFilePath,MNI_t1_brain,normPath)
inputDir <- '~/Data/ADNI/2Yr_1.5T'
outputFilesVector <- list.files(normPath)
inputFilesVector <- list.files(inputDir, recursive = TRUE)
inputFilesVector <- paste0(inputDir,"/",inputFilesVector)
inputFilesVector <- removeSubset(inputFilesVector, outputFilesVector)

lapply(inputFilesVector, normalise_MIN152, MNI_t1_brain, normPath)
