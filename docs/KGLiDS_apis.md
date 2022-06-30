## KGLiDS APIs
KGLiDS provides predefined operations in form of python apis that allow seamless integration with a
conventional data science pipeline.

<b>List of all APIs available:</b>

| S.no | API                                         | Description                                                                                                                                                                                                                     |
|------|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.   | `query()`                                   | Executes ad-hoc queries on fly                                                                                                                                                                                                  |
| 2.   | `show_graph_info()`                         | Summarizes the information captured by KGLiDS. Shows:<br/>1. Total number of datasets abstracted<br/>2. Total number of tables abstracted<br/>3. Total number of columns abstracted<br/>4. Total number of pipelines abstracted |
| 3.   | `get_datasets_info()`                       | Shows the number of tables and pipelines per dataset                                                                                                                                                                            |
| 4.   | `get_tables_info()`                         | Shows all tables alongside their physical file path and dataset                                                                                                                                                                 |
| 5.   | `search_tables_on()`                        | Searches tables containing specific column names.                                                                                                                                                                               |
| 6.   | `recommend_k_unionable_tables()`            | Returns the top k tables that are unionable                                                                                                                                                                                     |
| 7.   | `recommend_k_joinable_tables()`             | Returns the top k tables that are joinable                                                                                                                                                                                      |
| 8.   | `get_path_between_tables()`                 | Visualizes the paths between a starting table and the target one                                                                                                                                                                |
| 9.   | `get_pipelines_info()`                      | Shows the following information for all pipeline:<br/>1. Pipeline name<br/>2. Dataset<br/>3. Author<br/>4. Date written on<br/>5. Number of votes<br/>6. Score                                                                  |
| 10.  | `get_most_recent_pipeline()`                | Returns the most recent pipeline                                                                                                                                                                                                |
| 11.  | `get_top_k_scoring_pipelines_for_dataset()` | Returns the top k pipeline with the highest score                                                                                                                                                                               |
| 12.  | `search_classifier()`                       | Shows all the classifiers used for a dataset                                                                                                                                                                                    |
| 13.  | `get_hyperparameters()`                     | Returns the hyperparameter values that were used for a given classifier                                                                                                                                                         |
| 14.  | `get_top_k_library_used()`                  | Visualizes the top-k libraries that were used overall or for a given dataset                                                                                                                                                    |


<br/>
<b>API examples:</b>

1. `kglids.query()`
```python
from api.api import KGLiDS
import pandas as pd

kglids = KGLiDS()

my_custom_query = """
SELECT ?source {
?source_id rdf:type    kglids:Source    ;
           schema:name ?source          . } """
kglids.query(my_custom_query)
```
|     | Source |
|-----|--------|
| 0.  | kaggle | 

<hr/>

2. `kglids.show_graph_info()`
```python
kglids.show_graph_info()
```
|     | Datasets | Tables | Columns | Pipelines |
|-----|----------|--------|---------|-----------|
| 0.  | 101      | 969    | 418     | 9502      |

<hr/>

3. `kglids.show_dataset_info()`
```python
kglids.show_dataset_info()
```


|      | Dataset                                           | Number_of_tables |
|------|---------------------------------------------------|------------------|
| 0	   | COVID-19 Corona Virus India Dataset               | 	8               |
| 1	   | COVID-19 Dataset                                  | 	6               |
| 2	   | COVID-19 Healthy Diet Dataset                     | 	5               |
| 3	   | COVID-19 Indonesia Dataset                        | 	1               |
| 4	   | COVID-19 World Vaccination Progress               | 	2               |
| ...  | 	...                                              | 	...             |
| 96	  | uciml.red-wine-quality-cortez-et-al-2009          | 	22              |
| 97	  | unitednations.international-greenhouse-gas-emi... | 	3               |
| 98	  | upadorprofzs.testes	                              | 8                |
| 99	  | vitaliymalcev.russian-passenger-air-service-20... | 	14              |
| 100  | 	ylchang.coffee-shop-sample-data-1113             | 	10              |

