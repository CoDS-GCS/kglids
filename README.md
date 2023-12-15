# KGLiDS - Linked Data Science Powered by Knowledge Graphs

![KGLiDS_architecture](docs/graphics/kglids_architecture.jpg)


In recent years, we have witnessed the growing interest from academia and industry in applying data science technologies to analyze large amounts of data. While in this process a myriad of artifcats (datasets, pipeline scripts, etc.) are created, there has so far been no systematic attempt to holistically collect and exploit all the knowledge and experiences that are implicitly contained in those artifacts. Instead, data scientists resort to recovering information and experience from colleagues or learn via trial and error. 
Hence, this paper presents a scalable system, KGLiDS, that employs machine learning and knowledge graph technologies to abstract and capture the semantics of data science artifacts and their connections. Based on this information KGLiDS enables a variety of downstream applications, such as data discovery and pipelines automation. 
Our comprehensive evaluation covers use cases in data discovery, data cleaning, transformation, and AutoML and shows that KGLiDS is significantly faster with a lower memory footprint as the state of the art while achieving comparable or better accuracy.

## Technical Report
Our technical report is available on [ArXiv](https://arxiv.org/abs/2303.02204) and includes more details on our system and interfaces is available

## Linked Data Science: Systems and Applications
To learn more about Linked Data Science and its applications, please watch Dr. Essam Mansour's talk at Waterloo DSG Seminar ([Here](https://www.youtube.com/watch?v=99wvN04C5fU)). 

## Installation
* Clone the `kglids` repo 
* Create `kglids` Conda environment (Python 3.8) and install pip requirements.
* Activate the `kglids` environment
```commandline
conda create -n kglids python=3.8 -y
conda activate kglids
pip install -r requirements.txt
```

## Quickstart
<p>
<b>Try the Sample <a href="https://colab.research.google.com/drive/1XbjJkppz5_nTufgnD53gEBzxyLYViGAi?usp=sharing" style="color: orange"> KGLiDS Colab notebook</a>
for a quick hands-on! </b>
</p>


<b>Generating the LiDS graph:</b>
* Add the data sources to [config.py](kg_governor/data_profiling/src/config.py):
```python
# sample configuration
# list of data sources to process
data_sources = [DataSource(name='benchmark',
                           path='/home/projects/sources/kaggle',
                           file_type='csv')]

```
* Run the [Data profiler](kg_governor/data_profiling/src/main.py)
```commandline
cd kg_governor/data_profiling/src/
python main.py
```
* Run the [Knowledge graph builder](kg_governor/knowledge_graph_construction/src/data_global_schema_builder.py) to generate the data_items graph 
```commandline/
cd kg_governor/knowledge_graph_construction/src/
python data_global_schema_builder.py
```
* Run the [Pipeline abstractor](kg_governor/pipeline_abstraction/pipelines_analysis.py) to generate the pipeline named graph(s)
```
cd kg_governor/pipeline_abstraction/
python pipelines_analysis.py
```
<hr>

<b>Uploading LiDS graph to the graph-engine (we recommend using [GraphDB](https://graphdb.ontotext.com/) ):</b>
Please see [populate_graphdb.py](storage/utils/populate_graphdb.py) for an example of uploading graphs to GraphDB.



<hr>

<b> Using the KGLiDS APIs</b>: 

KGLiDS provides predefined operations in form of python apis that allow seamless integration with a conventional data science pipeline.
Checkout the full list of [KGLiDS APIs](docs/KGLiDS_apis.md)

## LiDS Ontology
To store the created knowledge graph in a standardized and well-structured way,
we developed an ontology for linked data science: the <b>LiDS Ontology</b>.<br/>
Checkout [LiDS Ontology](docs/LiDS_ontology.md)!

## Benchmarks
The following benchmark datasets were used to evaluate KGLiDS:
* Data Discovery: Table Union Search
  * [D<sup>3</sup>L Small](https://github.com/alex-bogatu/d3l)
  * [TUS Small](https://github.com/RJMillerLab/table-union-search-benchmark)
  * [SANTOS Small](https://github.com/northeastern-datalab/santos/tree/main)
  * [SANTOS Large](https://github.com/northeastern-datalab/santos/tree/main) 

* [Data Cleaning, Data Transformation, and AutoML](gnn_applications/README.md)
* Kaggle
  * [`setup_kaggle_data.py`](storage/utils/setup_kaggle_data.py)

## KGLiDS APIs
See the full list of supported APIs [here](docs/KGLiDS_apis.md).

## Citing Our Work
If you find our work useful, please cite it in your research.
```
@article{kglids,
         title={Linked Data Science Powered by Knowledge Graphs}, 
         author={Mossad Helali and Shubham Vashisth and Philippe Carrier and Katja Hose and Essam Mansour},
         year={2023},
         journal={ArXiv},
         url = {https://arxiv.org/abs/2303.02204}
}
```


## Contributions
We encourage contributions and bug fixes, please don't hesitate to open a PR or create an issue if you face any bugs.

## Questions
For any questions please contact us:

mossad.helali@mail.concordia.ca

essam.mansour@concordia.ca
