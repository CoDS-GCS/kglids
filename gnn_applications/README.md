# Graph Neural Network (GNN) Applications
This directory contains the GNN-based applications of KGLiDS, including Data Cleaning, Data Transformation, and AutoML.

# Benchmark Datasets
The following table contains the names and details of benchmark datasets used to evaluate the different GNN applications of KGLiDS:

|Num|Dataset                               |Size (MB)|#Rows  |#Columns|#Categorical|#Numerical|%MV     |ML Task        |
|---|--------------------------------------|---------|-------|--------|------------|----------|--------|---------------|
|1  |hepatitis                             |0.01     |155    |20      |0           |20        |0.0539  |Classification |
|2  |housevotes84                          |0.02     |435    |17      |17          |0         |0.0530  |Classification |
|3  |horsecolic                            |0.02     |300    |28      |0           |28        |0.1911  |Classification |
|4  |breastcancerwisconsin                 |0.02     |699    |11      |0           |11        |0.0021  |Classification |
|5  |credit                                |0.03     |690    |16      |9           |7         |0.0061  |Classification |
|6  |cleveland_heart_disease               |0.06     |303    |15      |0           |15        |0.0013  |Classification |
|7  |titanic                               |0.08     |891    |12      |5           |7         |0.0902  |Classification |
|8  |credit-g                              |0.16     |1000   |21      |14          |7         |0.0037  |Classification |
|9  |jm1                                   |1.75     |10885  |21      |0           |21        |0.0001  |Classification |
|10 |adult                                 |5.59     |48842  |15      |9           |6         |0.0089  |Classification |
|11 |higgs                                 |21.69    |98050  |29      |0           |29        |0       |Classification |
|12 |APSFailure                            |99.15    |76000  |171     |1           |170       |0.0830  |Classification |
|13 |albert                                |256.30   |425240 |79      |0           |79        |0.1346  |Classification |
|13 |Fertility Diagnosis                   |0.0125   |100    |10      |9           |1         |0       |Classification |
|14 |haberman                              |0.0095   |306    |4       |4           |0         |0       |Classification |
|15 |wine                                  |0.0191   |178    |14      |14          |0         |0       |Classification |
|16 |ecoli                                 |0.0584   |336    |9       |7           |2         |0       |Classification |
|17 |Pima Indians Subset                   |0.0529   |768    |9       |9           |0         |0       |Classification |
|18 |dermatology                           |0.1157   |366    |35      |34          |1         |0.0006  |Classification |
|19 |banknote authentication               |0.0525   |1372   |5       |5           |0         |0       |Classification |
|20 |Ionosphere                            |0.0939   |351    |35      |35          |0         |0       |Classification |
|21 |Sonar                                 |0.0969   |208    |61      |61          |0         |0       |Classification |
|22 |abalone                               |0.4861   |4177   |9       |8           |1         |0       |Classification |
|23 |libras                                |0.2501   |360    |91      |91          |0         |0       |Classification |
|24 |waveform                              |0.8394   |5000   |22      |22          |0         |0       |Classification |
|25 |letter recognition                    |3.5478   |20000  |17      |16          |1         |0       |Classification |
|26 |optical digits                        |2.7872   |5620   |65      |65          |0         |0       |Classification |
|27 |feature pixel                         |3.6775   |2000   |241     |241         |0         |0       |Classification |
|28 |shuttle                               |4.4253   |58001  |10      |10          |0         |0       |Classification |
|29 |feature Fourier                       |1.1751   |2000   |77      |77          |0         |0       |Classification |
|30 |poker                                 |86.0224  |1025010|11      |11          |0         |0       |Classification |
|31 |airlines                              |18.2766  |539383 |8       |3           |5         |0       |binary         |
|32 |blood-transfusion-service-center      |0.01     |748    |5       |0           |5         |0       |binary         |
|33 |christine                             |31.3888  |5418   |1637    |0           |1637      |0       |binary         |
|34 |kc1                                   |0.1446   |2109   |22      |0           |22        |0       |binary         |
|35 |MiniBooNE                             |69.3548  |130064 |51      |0           |51        |0       |binary         |
|36 |sylvine                               |0.4107   |5124   |21      |0           |21        |0       |binary         |
|37 |cnae-9                                |1.771    |1080   |857     |0           |857       |0       |multi-class    |
|38 |connect-4                             |5.541    |67557  |43      |0           |43        |0       |multi-class    |
|39 |dilbert                               |175.9978 |10000  |2001    |0           |2001      |0       |multi-class    |
|40 |fabert                                |13.0473  |8237   |801     |0           |801       |0       |multi-class    |
|41 |jungle_chess_2pcs_raw_endgame_complete|0.5985   |44819  |7       |1           |6         |0       |multi-class    |
|42 |robert                                |268.0512 |10000  |7201    |0           |7201      |0       |multi-class    |
|43 |vehicle                               |0.0529   |846    |19      |1           |18        |0       |multi-class    |
|44 |fri_c1_1000_25                        |0.2259   |1000   |26      |1           |25        |0       |binary         |
|45 |car_evaluation                        |0.0728   |1728   |22      |0           |22        |0       |multi-class    |
|46 |glass                                 |0.0101   |205    |10      |0           |10        |0       |multi-class    |
|47 |kropt                                 |0.5073   |28056  |7       |4           |3         |0       |multi-class    |
|48 |cpu_act_761                           |0.6856   |8192   |22      |1           |21        |0       |binary         |
|49 |cpu_small_735                         |0.426    |8192   |13      |1           |12        |0       |binary         |
|50 |page-blocks                           |0.2256   |5473   |11      |1           |10        |0       |binary         |
|51 |waveform-5000                         |1.0052   |5000   |41      |1           |40        |0       |binary         |
