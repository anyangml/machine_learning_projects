# Project_Chemicals

## Abstract

In the field of cheminformatics, researchers have a great interest in using molecular structures to predict the corresponding molecular properties. There are many ways to describe molecules in a way that computers can understand, among which SMILES is a widely used encoding method. "SMILES" is an acronym for Simplified Molecular Input Line System and they use short ASCII strings to describe the structure of chemical species. Extracting molecular properties from SMILES strings is a common task in cheminformatics. In this project, we demonstrate a simplified workflow and showcase some widely used toolkits in cheminformatics.

The overall goal of _Project_chemicals_ is to use the SMILES string to predict the boiling point of a given molecule, and the workflow is shown below.

        Obtain Data  -->  Clean Data  -->  Explore Data  -->  Build Model  -->  Evaluate and Interpret Model

## Results and Discussion

### Scraping Data

We want a dataset that has both SMILES strings and boiling points. The closest dataset I can easily get is the CRC Handbook, which contains boiling points and CAS numbers, but it does not have SMILES strings. We need to use the "cirpy" toolbox to convert the CAS numbers to SMILES strings. _The dataScraper()_ funtion scrapes the raw data and the _dataCleaner()_ function removes unwanted data and performs the CAS number-to-SMILES string conversion. The pre-processed data is then saved in a .csv file for easy access.

#### Data Source

<p align="center">
  <img width="1487" alt="image" src="https://user-images.githubusercontent.com/66216181/176576603-703ced8a-28cd-44a3-a50d-7a8ec2099395.png">

</p>

#### Scraped Data

<p align="center">
  <img width="858" alt="image" src="https://user-images.githubusercontent.com/66216181/176576845-39967674-57f3-4a25-808c-077c625a908b.png">

</p>

### Validating Data

We want to check if the boiling point has a skewed distribution and to check if the SMILES strings are correct. So we plot a histgram using the boiling and we visualize the molucules with their SMILES strings. It seems the boilings are roughly normally distributed and the SMILES strings are correct.

<p align="center">
  <img width="388" alt="image" src="https://user-images.githubusercontent.com/66216181/176577634-e76f7e43-8275-407d-8245-cbca2a46b4a7.png">
</p>
<p align="center">
  <img width="603" alt="image" src="https://user-images.githubusercontent.com/66216181/176577713-9acca6ec-d003-49dd-b4bf-ea73ab1516d5.png">

</p>

### Feature Engineering

We then generate quantitative features to describe the molecules using their SMIELS strings, 'mordred' is a package does just that. The calculated features are saved in a .csv file for easy access. Only features with top correlation scores are used for demonstration.

<p align="center">
  <img width="337" alt="image" src="https://user-images.githubusercontent.com/66216181/176578096-94437a28-ecc9-422e-94a2-62c6c6781384.png">

</p>

### Training and Interpreting Model

Several descriptors are chosen to predict the target value. Two different tree ensembles are demonstrated here, the RandomForest regressor using bagging method and the XGBoost Regressor using gradient boosting method. From the parity plot, we can see that both algorithms can make reasonable predictions of boiling points. The dataset is very small, so XGBoost does not significantly outperform RandomForest. to understand the importance of the features in the model, we can use a package called "SHAP".

<p align="center">
  <img width="344" alt="image" src="https://user-images.githubusercontent.com/66216181/176578406-9df024cb-13ab-4bd0-b90d-78d68011be2a.png">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/66216181/176578480-1999ed45-1b0d-4f6b-aeae-2ca90b9531e0.png">

</p>

The Shap analysis shows that _piPC1, TpiPC10, nC_ are the three most important features. Features that have high correlation with target value may not be important due to Multicollinearity.
