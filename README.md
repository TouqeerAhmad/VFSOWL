# VFSOWL

Implementation for our CVPR workshop paper listed below:  

**[Variable Few Shot Class Incremental and Open World Learning](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Ahmad_Variable_Few_Shot_Class_Incremental_and_Open_World_Learning_CVPRW_2022_paper.html)**, [CVPR-Workshops 2022.](https://cvpr2022.thecvf.com/)

Authors: [Touqeer Ahmad](https://sites.google.com/site/touqeerahmadsite/Touqeer?authuser=0), [Akshay Raj Dhamija](https://akshay-raj-dhamija.github.io/), [Mohsen Jafarzadeh](http://www.mohsen-jafarzadeh.com/index.php), [Steve Cruz](https://scholar.google.com/citations?user=_zl-yoMAAAAJ&hl=en), [Ryan Rabinowitz](https://scholar.google.com/citations?hl=en&user=w-3eXsMAAAAJ), [Chunchun Li](https://scholar.google.com/citations?user=xPJiRT0AAAAJ&hl=en), and [Terrance E. Boult](https://vast.uccs.edu/~tboult/) 

The paper is focused on Variable Few-Shot Class Class Incremental (VFSCIL) and Variable Few-Shot Open-World Learning (VFSOWL). Unlike earlier approaches on Few-Shot Class Incremental Learning (FSCIL) that assume fixed number of classes (N-ways) and fixed number of samples (K-shots), VFSCIL operates in a more natural/practical setting where each incremental session could have up-to-N-classes (ways) and each class could have up-to-K-samples (shots). VFSCIL is then extended into VFSOWL.

The approach extended for VFSCIL/VFSOWL stems from our concurrent work on FSCIL named [FeSSSS](https://github.com/TouqeerAhmad/FeSSSS).

## Variable Few-Shot Class Incremental Learning
Here we focus the description for CUB200 dataset, similar details follow for miniImageNet. First, all the incremental session files are generated using the script ```random_N_Ways_K_Shots_cub200.py``` for different experimental settings. We have explored the following four experimental settings: 

* Up-to 10-Ways, Up-to 10-Shots (15 incremental sessions)
* Up-to 10-Ways, Up-to 5-Shots (15 incremental sessions)
* Up-to 5-Ways, Up-to 5-Shots (30 incremental sessions)
* Up-to 5-Ways, Up-to 10-Shots (30 incremental sessions)

For each experimental setting, we generate 5 experiments and those session files are made available in respective directories inside experiments_cub200 directory for using exactly the same instances as we used in our experiments. The base session is still comprised of 100 classes and 30 samples-per-class. The instances for base session are identical to earlier work on fixed-FSCIL e.g., [CEC](https://github.com/icoz69/CEC-CVPR2021). More experiments for the said experimental settings, or even different experimental settings can be generated using the above stand-alone code file by altering the number of increments and N_ways/K_shots accordingly.       






### BibTeX
If you find our work helpful, please cite the following:

```
@InProceedings{Ahmad_2022_CVPR,
    author    = {Ahmad, Touqeer and Dhamija, Akshay Raj and Jafarzadeh, Mohsen and Cruz, Steve and Rabinowitz, Ryan and Li, Chunchun and Boult, Terrance E.},
    title     = {Variable Few Shot Class Incremental and Open World Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {3688-3699}
}
``` 
