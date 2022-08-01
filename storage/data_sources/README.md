Input datasets and pipelines are placed under this directory.

They should have the following structure for each data source:

```
<SOURCE_NAME>
    |__ <DATASET_1_NAME>
            |__ tables
            |       |__ table1.csv
            |       |__ table2.csv
            |
            |__ notebooks
                    |__ notebook1.ipynb
                    |__ notebook2.py
```

Example:

```
kaggle
    |__ titanic
    |       |__ tables
    |       |       |__ train.csv
    |       |       |__ test.csv
    |       |
    |       |__ notebooks
    |               |__ my_first_pipeline.ipynb
    |               |__ regression.py
    |
    |__ soccer_database
            |__ tables
            |       |__ dataset.csv
            |
            |__ notebooks
                    |__ pipeline1.ipynb
                    |__ trying_out_classifiers.py

```