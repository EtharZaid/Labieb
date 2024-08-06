import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

'''
This code takes in an excel file with multiple samples with the following fields:
	Model_Response: The field extracted by the model
	Ground_Truth: The correct value
	Comparison_Score: 0 or 1 (1 if the model was correct or 0 otherwise)
	Final_Confidence: 0-1 value of the estimated confidence

The code plot the ROC along with the AUC value, where higher values (closer to 1) shows
that the estimated confidence is actually indicative of the accuracy

The code then calculates both Kappa score and Acuracy 
'''


#Prepare the figure
fig,ax=plt.subplots()

Responses=pd.read_excel(r'/path/to/excel.xlsx')
Responses.fillna('NA',inplace=True)

#ROC evaluates wether the confidence is reflective of the correctness of the model
Labels=np.array(Responses['Comparison_Score'] )
Confidence=np.array(Responses['Final_Confidence'] )


RocCurveDisplay.from_predictions(
    Labels,
    Confidence,
    name="GPT's Performance",
    color="navy",
    linewidth=2,
    ax=ax,
)

plt.axis("square")
plt.xlabel("False Positive Rate",fontsize=14)
plt.ylabel("True Positive Rate",fontsize=14)
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.show()


#Kappa statistics calculation to measure the agreement between the model extraction 
#and the true labels

GroundTruth= Responses['Comparison_Score']
ModelResponse= Responses['Model_Response']


kappa = cohen_kappa_score(GroundTruth, ModelResponse)

print('Kappa score = ',kappa)

#Calculate Accuracy

correct_predictions = GroundTruth == ModelResponse
correct_predictions = correct_predictions.astype(int)
accuracy = correct_predictions.mean()
print('Accuracy = ',accuracy)

