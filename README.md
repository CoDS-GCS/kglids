# KGLiDS - A Knowledge Graph-Based Platform for Linked Data Science

![alt text](docs/graphics/kglids_architecture.jpg)


<div style="text-align: justify">In recent years, we have witnessed a growing interest in data science
not only from academia but particularly from companies investing
in data science platforms to analyze large amounts of data. In this
process, a myriad of data science artifacts, such as datasets and
pipeline scripts, are created. Yet, there has so far been no systematic
attempt to holistically exploit the collected knowledge and expe-
riences that are implicitly contained in the specification of these
pipelines, e.g., compatible datasets, cleansing steps, ML algorithms,
parameters, etc. Instead, data scientists still spend a considerable
amount of their time trying to recover relevant information and
experiences from colleagues, trial and error, lengthy exploration,
etc. In this paper, we therefore propose a novel system (KGLiDS)
that employs machine learning to extract the semantics of data
science pipelines and captures them in a knowledge graph, which
can then be exploited to assist data scientists in various ways. This
abstraction is the key to enable Linked Data Science since it allows
us to share the essence of pipelines between platforms, companies,
and institutions without revealing critical internal information and
instead focusing on the semantics of what is being processed and
how. Our comprehensive evaluation uses thousands of datasets and
more than thirteen thousand pipeline scripts extracted from data
discovery benchmarks and the Kaggle portal, and show that KGLiDS
significantly outperforms state-of-the-art systems on related tasks,
such as datasets and pipeline recommendation.</div>

## Installation
* Clone the `kglids` repo 
* Create `kglids` Conda environment (Python 3.8) and install pip requirements.
* Activate the `kglids` environment
```
conda activate kglids
```

## Quickstart
<b>Try the [Sample Colab notebook](https://colab.research.google.com/drive/1XbjJkppz5_nTufgnD53gEBzxyLYViGAi?usp=sharing) for a quick hands-on!</b><hr>

<b>Generating the LiDS graph:</b>
1. Add your configurations to [config,yml]():
```python
# sample configurations
datasource: "kaggle" 
datasets:
- name:     "Titanic"
  type:     "csv"
  path:     "/data/titanic"
  origin:   "https://www.kaggle.com/"
- name:     "Montreal Crime Data"
  type:     "csv"
  path:     "/data/mtl_crime"
  origin:   "https://www.kaggle.com/"
```
</t>Note: The config.yml file expects the following structure:
```
datasource/
├── datasets
│   └── name
│   └── type
│   └── path
│   └── origin
├ ...
...
```
2. Run the [Data profiler]()
```commandline
cd kglids/data_items/profiler/src/
python main.py
```
3. Run the [Knowledge graph builder]() to generate the data_items graph 
```commandline/
cd kglids/data_items/knowledge_graph/src/
python knowledge_graph_builder.py
```
4. Run the [Pipeline abstractor]() to generate the pipeline named graph(s)
```
cd kglids/pipelines/src/
python run.py
```
<hr>

<b>Uploading LiDS graph to the graph-engine (we recommend using [Stardog](https://www.stardog.com/)):</b>
1. Create a database 
Note: enable support for <i>RDF *</i> (example given below) more info [here](https://docs.stardog.com/query-stardog/edge-properties)
```python
import stardog
database_name = 'db'
connection_details = {
      'endpoint': 'http://localhost:5820',
      'username': 'admin',
      'password': 'admin'}
with stardog.Admin(**connection_details) as admin:
    if database_name in [db.name for db in admin.databases()]:
        admin.database(database_name).drop()
    db = admin.new_database(database_name, {'edge.properties': True})
```
2. Add the dataset-graph to the database
```
stardog data add --format turtle db dataset_graph.nq
```
3. Add the pipeline default graph and named-graphs to the database
```
stardog data add --format turtle db dataset_graph.nq
```
```python
conn = stardog.Connection(database_name, **connection_details)
conn.begin()
ttl_files = [i for i in os.listdir(graphs_dir) if i.endswith('ttl')]
for ttl in ttl_files:
    conn.add(stardog.content.File(graphs_dir + ttl), graph_uri= 'http://kglids.org/pipelineResource/'
conn.commit()
conn.close()
```
<hr>

<b> Using the KGLiDS APIs</b>: 

KGLiDS provides predefined operations in form of python apis that allow seamless integration with a conventional data science pipeline.
Checkout the full list of [KGLiDS APIs](docs/KGLiDS_apis.md)

## LiDS Ontology
To store the created knowledge graph in a standardized and well-structured way,
we developed an ontology for linked data science: the <b>LiDS Ontology</b>.<br/>
Checkout [LiDS Ontology](docs/LiDS_ontology.md)!

## Benchmarking
The following benchmark datasets were used to evaluate KGLiDS:
* Dataset Discovery in Data Lakes
  * [Smaller Real](https://github.com/alex-bogatu/d3l)
  * [Synthetic](https://github.com/RJMillerLab/table-union-search-benchmark)<br>
    (more info on data discovery benchmarks [here]((https://arxiv.org/pdf/2011.10427.pdf))) 
* Kaggle
  * [`setup_kaggle_data.py`]()

## KGLiDS APIs
See the full list of supported APIs [here](docs/KGLiDS_apis.md).

## Citing Our Work
If you find our work useful, please cite it in your research:

## Questions
For any questions please contact us at:<br/>mossad.helali@concordia.ca, shubham.vashisth@concordia.ca, philippe.carrier@concordia.ca, khose@cs.aau.dk, essam.mansour@concordia.ca
