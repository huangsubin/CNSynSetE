# CNSynSetE
Paper title: A Bilateral Context and Filtering Strategy based Approach to Chinese Entity Synonym Set Expansion  
######################################################################################  
This resource includes datasets, source code (training,evaluation, and prediction) for the paper: [A Bilateral Context and Filtering Strategy-based Approach to Chinese Entity Synonym Set Expansion].  
  
Details about the resource, such as datasets and source code, are available at (http://cslab.ahpu.edu.cn/CNSynSetExpan/).  
 
######################################################################################  
# Dependencies  
 Python 3 with NumPy  
 gensim  
 sklearn  
 scikit-opt  
 PyTorch  
 tqdm  
 tensorboardX  
 networkx  
######################################################################################  
# Training  
 ## For SoGouCA(SGSynSetTra)  
 ### Setting the [ini_parameters.ini] as follows:     
 [ini_parameters_mode]  
 ##mode of the project: train,eval_opt, and test  
 mode = train  
 
   
 [ini_parameters_train]  
 ##train on RTX6000 with GPU-RAM=24GB  
 ##dataset: training the model for BaiDu or SoGouCA respectively   
 ##The trained models are saved in runs->snapshots_trianed_best->SoGouCA  or runs->snapshots_trianed_best->Baidu  
 ##The trained results are saved in runs->training_log_mu0.0_SoGouCA  or runs->training_log_mu0.0_BaiDu  
######################################################################################  

 ##following is the ini_parameters_train for  SoGouCA  
 dataset=SoGouCA  

 ##mu:  using 0.0-0.9 to train the model respectively.  
 mu=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  

 ##delta: default=0.2  other values will evaluate in the eval_opt step  
 delta=0.2  
  
 ##kappa:default=0.5  other values will evaluate in the eval_opt step  
 kappa=0.5  
  
 ##lamb:default=0.5  other values will evaluate in the eval_opt step  
 lamb=0.5  
### Runing the script main.py    
## For BaiDu(BDSynSetTra)  
### Setting the [ini_parameters.ini] as follows:  
[ini_parameters_mode]   
##mode of the project: train, eval_opt, and test  
mode = train  
  
    
[ini_parameters_train]  
##train on RTX6000 with GPU-RAM=24GB  
##dataset: training the model for BaiDu or SoGouCA respectively   
##The trained models are saved in runs->snapshots_trianed_best->SoGouCA  or runs->snapshots_trianed_best->Baidu  
##The trained results are saved in runs->training_log_mu0.0_SoGouCA  or runs->training_log_mu0.0_BaiDu  
######################################################################################  

##following is the ini_parameters_train for  BaiDu  
dataset=BaiDu  
  
##mu:  using 0.0-0.9 to train the model respectively  
mu=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  
  
##delta:default=0.2 other values will evaluate in the eval_opt step  
delta=0.2  
  
##kappa:default=0.6 other values will evaluate in the eval_opt step  
kappa=0.6   
  
##lamb:default=0.5 other values will evaluate in the eval_opt step   
lamb=0.5  
### Runing the script main.py  
######################################################################################  
  
# Evaluation  
## For SoGouCA(SGSynSetTra)    
### Setting the [ini_parameters.ini] as follows:  
[ini_parameters_mode]  
##mode of the project: train, eval_opt, and test  
mode = eval_opt  
  
[ini_parameters_evaluation_opt]  
##Using the PSO to optimal the best evaluation parameters for the project  
##The evaluation parameter results are saved in runs->evaluation_log_BaiDu  or runs->evaluation_log_SoGouCA  

######################################################################################  
##following is the ini_parameters_evaluation_opt for SoGouCA  
  
###mutil_processNum, type=int; the number of the parallel processes.  1 denotes no parallel computing,  beger than 1 denotes parallel computing  
###For SoGouCA dataset, mutil_processNum=8, GPU-RAM=24GB; For Baidu dataset, mutil_processNum=3, GPU-RAM=24GB  
mutil_processNum=8  
  
##dataset, BaiDu or SoGouCA  
dataset=SoGouCA  
  
##mu:  the lp and up limit of the mu  
mu=[0.1,0.9]  
  
##delta: the lp and up limit of the delta  
delta=[0.1,0.9]  
  
##kappa: the lp and up limit of the kappa  
kappa=[0.1,0.9]  
  
##lamb: the lp and up limit of the lamb  
lamb=[0.1,0.9]  
  
### Runing the script main.py  
  
## For BaiDu(BDSynSetTra)  
### Setting the [ini_parameters.ini] as follows:  
[ini_parameters_mode]  
##mode of the project: train, eval_opt, and test  
mode = eval_opt  
  
[ini_parameters_evaluation_opt]  
##Using the PSO to optimal the best evaluation parameters for the project  
##The evaluation parameter results are saved in runs->evaluation_log_BaiDu  or runs->evaluation_log_SoGouCA  
    
######################################################################################  
##following is the ini_parameters_evaluation_opt for BaiDu  
  
###mutil_processNum, type=int; the number of the parallel processes.  1 denotes no parallel computing,  beger than 1 denotes parallel computing  
###For SoGouCA dataset, mutil_processNum=8, GPU-RAM=24GB; For Baidu dataset, mutil_processNum=3, GPU-RAM=24GB  
mutil_processNum=3  
  
##dataset, BaiDu or SoGouCA  
dataset=BaiDu  
  
##mu:  the lp and up limit of the mu  
mu=[0.1,0.9]  
  
##delta: the lp and up limit of the delta  
delta=[0.1,0.9]  
  
##kappa: the lp and up limit of the kappa  
kappa=[0.1,0.9]  
  
##lamb: the lp and up limit of the lamb  
lamb=[0.1,0.9]  

### Runing the script main.py  
######################################################################################  
  
# Prediction  
## For SoGouCA(SGSynSetTra)  
### Setting the [ini_parameters.ini] as follows:  
[ini_parameters_mode]  
##mode of the project: train, eval_opt, and test  
mode = test  
  
[ini_parameters_test]  
##Predicting the project using the best parameters or other parameters  
##The prediction results are saved in runs->testing_log_BaiDu  or runs->testing_log_SoGouCA  
######################################################################################  
##following is the ini_parameters_test for SoGouCA  
  
##dataset: Predicting the model for BaiDu or SoGouCA respectively  
dataset=SoGouCA  
  
##mu:  chooseing one of a figure or a list from 0.0-0.9 to test the project  
mu=[0.2]  
  
##delta: chooseing one of a figure from 0.0-0.9 to test the project  
delta=[0.1]  
  
##kappa: chooseing one of a figure from 0.0-0.9 to test the project  
kappa=[0.1]  
  
##lamb: chooseing one of a figure from 0.0-0.9 to test the project  
lamb=[0.6]  
  
### Runing the script main.py
  
## For BaiDu(BDSynSetTra)  
### Setting the [ini_parameters.ini] as follows:  
[ini_parameters_mode]  
##mode of the project: train,eval_opt, and test  
mode = test  
  
[ini_parameters_test]  
##Predicting the project using the best parameters or other parameters  
##The prediction results are saved in runs->testing_log_BaiDu  or runs->testing_log_SoGouCA  
  
######################################################################################  
##following is the ini_parameters_test for BaiDu  

##dataset: Predicting the model for BaiDu or SoGouCA respectively  
dataset=BaiDu  
  
##mu:  chooseing one of a figure from 0.0-0.9 to test the project  
mu=[0.3]  
  
##delta: chooseing one of a figure from 0.0-0.9 to test the project  
delta=[0.7]  
  
##kappa: chooseing one of a figure from 0.0-0.9 to test the project  
kappa=[0.7]  
  
##lamb: chooseing one of a figure from 0.0-0.9 to test the project  
lamb=[0.2]  
  
## Runing the script main.py
  
  
# References
  
If our paper and the above datasets and source code are useful for your research, please cite the following paper in your publication:  
  
@article{Huang2022CNSynSetE,  
 title={A Bilateral Context and Filtering Strategy based Approach to Chinese Entity Synonym Set Expansion},  
 author={Subin Huang and Yu Xiu and Jun Li and Sanmin Liu and Chao Kong},  
 journal={Complex & Intelligent Systems},  
 volume={*},  
 pages={*},  
 year={2023 accept} 
 }
