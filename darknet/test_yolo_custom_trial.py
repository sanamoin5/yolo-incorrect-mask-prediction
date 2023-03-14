import re
from subprocess import call, Popen, PIPE
import pandas as pd
import sys
import time
import subprocess
import os

sys.stdout = open('res_preds.txt','w+')
f = open('../data/pwmfd/val.txt','r').read().split('\n')
l = ['filename','classes','detection %']
df = pd.DataFrame(columns=l)
df2 = pd.DataFrame(columns=l)

f_write = open('df_write_output.txt','w+')

for file in f:
    # process = Popen(['./darknet', 'detector', 'test', '../config/pwmfd.data', '../config/pwmfd.cfg', '../backup/pwmfd_final.weights',file,'>res.txt'])
    if file:
        time.sleep(1)
        process = os.system('./darknet detector test ../config/pwmfd.data ../config/pwmfd.cfg ../backup/pwmfd_final.weights '+file+' >res.txt')
        time.sleep(9)
        f_out = open('res.txt','r').read().split('\n')        
        for i,data in enumerate(f_out):
            txt1 = re.findall(r'(.*_mask)\: (\d\d?\d?)\%', str(data))
            fname = file.split('/')[-1]
            if len(txt1)>0:
                df = df.append({'filename':fname, 'classes':txt1[0][0], 'detection %':txt1[0][1]}, ignore_index=True)
                df2 = df2.append({'filename':file, 'classes':txt1[0][0], 'detection %':txt1[0][1]}, ignore_index=True)
                f_write.write(file+' '+txt1[0][0]+' '+txt1[0][1]+'\n')
                os.system('mv predictions.jpg out_predictions/predictions_'+fname)

sys.stdout.close()
df.to_csv('res_val_pred.csv')
df2.to_csv('res_val_pred_filepath.csv')
