import matplotlib.pyplot as plt
import random
import os
import numpy as np
import pandas as pd
from path import RESULT_PATH

rootdir = RESULT_PATH+'AUC_data/'
pathList = []
folderList = []
aucList = []
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        pathList.append(d)
        folderList.append(file)

# folderList.sort()
for i in range(len(folderList)):
    protein = folderList[i]
    proteinFolder = pathList[i]
    files = os.listdir(proteinFolder+'/')
    # print(files)
    for f in files:
        if f != 'all_measures.txt':
            pathToFile = os.path.join(proteinFolder+'/', f)
            # print(pathToFile)
            txt_file = open(pathToFile, "r")
            file_content = txt_file.read()
            file_content = file_content.replace("[", "")
            file_content = file_content.replace("]", "")
            file_content = file_content.replace("\n", "")
            content_list = file_content.split(" ")
            txt_file.close()
            content_list = list(filter(None,content_list))
            values = [float(i) for i in content_list]

            if f == 'mean_tpr.txt':
                mean_tpr = values
            elif f == 'mean_fpr.txt':
                mean_fpr = values
            elif f == 'AUC.txt':
                auc = values[0]
                aucList.append(auc)
    random_color= (random.random(), random.random(), random.random())
    # print(protein+' %0.4f' % auc)
    plt.plot(mean_fpr, mean_tpr, color=random_color, lw=0.7, label= protein+':%0.4f' % auc)


dataSheet = pd.DataFrame({'Protein':folderList,'Mean AUC':aucList})
dataSheet.to_excel(rootdir+'../aucs.xlsx', sheet_name='sheet1', index=False)

print('No of datasets : ',len(folderList))
print('Mean AUC : ',np.mean(aucList))
plt.plot([0, 1], [0, 1], color='navy', lw=0.7, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(ncol=2, loc="lower right",prop={'size': 7})
plt.savefig(rootdir+'../roc.png', dpi=300)
plt.show()