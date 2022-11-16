# Dragon WP8: detection of COVID-19 severity


:wave: A <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"> ***Random Forest classifier***</a> is applied on the clinical and demographics data to identify the severity of COVID-19. 


## Evaluation and Results:
10-fold cross validation was used for the model evaluation. We computed the Area Under the Receiver Operating Characteristic Curve to evaluate the classification performance.
AUC = 0.90 +/- 0.01


### Feature importance based on feature permutation (for more information see <a href="https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance">this link</a>):

![](https://github.com/Nastaranrad/Dragon/blob/main/pics/FeatureImportance.png)

### Python package requirements:

* Python 3.9
* Scikit-learn 1.0.2
* Numpy 1.21.5
* pyreadstat 1.1.9

### Instruction for running the script:
* Set the path to the original data (.SAV format) in the <a href=https://github.com/Nastaranrad/Dragon/blob/main/clinical_data_analysis.py>***clinical_data_analysis***</a>
* For multi-class classification, you need to change ***y*** in the <a href=https://github.com/Nastaranrad/Dragon/blob/main/clinical_data_analysis.py>***clinical_data_analysis***</a> with the corresponding multi-class labels.
