# KGLiDS - A Knowledge Graph-Based Platform for Linked Data Science

n recent years, we have witnessed a growing interest in data science
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
such as datasets and pipeline recommendation.

## Installation
* Create `kglids` Conda environment (Python 3.8) and install pip requirements.
* Activate the `kglids` environment
```
conda activate kglids
```

## Quickstart
<b>Try the [Sample Colab notebook](https://colab.research.google.com/drive/1XbjJkppz5_nTufgnD53gEBzxyLYViGAi?usp=sharing) for a quick hands-on!</b>
- Generate the `LiDS` graph
- Upload `LiDS` graph to the [Stardog](https://www.stardog.com/) server
- Test the [`KGLiDS APIs`]()

## Benchmarking
The following benchmark datasets were used to evaluate KGLiDS:
* [Dataset Discovery in Data Lakes](https://arxiv.org/pdf/2011.10427.pdf)
* [GraphGen4Code]()
* [Kaggle]()


## LiDS Ontology

<section id="metadata">
    <h4 style="display:none;">Metadata</h4>
    <dl>
        <dt>URI</dt>
        <dd><code>http://kglids.org/ontology/</code></dd>
        <dt>Ontology RDF</dt>
        <dd><a href="data_file.ttl">RDF (turtle)</a></dd>
    </dl>
</section>
<section id="toc">
    <h4>Table of Contents</h4>
    <ol>
        <li><a href="#classes">Classes</a></li>
        <li><a href="#objectproperties">Object Properties</a></li>
        <li><a href="#datatypeproperties">Datatype Properties</a></li>
        <li><a href="#namespaces">Namespaces</a></li>
        <li><a href="#legend">Legend</a></li>
    </ol>
</section>
  <section id="classes">
    <h4>Classes <span style="float:right; font-size:smaller;"><a href="">&uparrow;</a></span></h4>
    <ul class="hlist">
        <li><a href="#API">API</a></li>
        <li><a href="#Class">Class</a></li>
        <li><a href="#Column">Column</a></li>
        <li><a href="#DataItem">DataItem</a></li>
        <li><a href="#DataScienceItem">DataScienceItem</a></li>
        <li><a href="#Dataset">Dataset</a></li>
        <li><a href="#Function">Function</a></li>
        <li><a href="#Library">Library</a></li>
        <li><a href="#Package">Package</a></li>
        <li><a href="#Pipeline">Pipeline</a></li>
        <li><a href="#PipelineItem">PipelineItem</a></li>
        <li><a href="#Source">Source</a></li>
        <li><a href="#Statement">Statement</a></li>
        <li><a href="#Table">Table</a></li>
    </ul>
    <div class="entity class" id="API">
        <h3>API<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/API</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/PipelineItem">PipelineItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>Sub-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/Function">Function</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/Package">Package</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/Class">Class</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/Library">Library</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsAPI">http://kglids.org/ontology/pipeline/callsAPI</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Class">
        <h3>Class<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Class</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/API">API</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsClass">http://kglids.org/ontology/pipeline/callsClass</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Column">
        <h3>Column<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Column</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/DataItem">DataItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In domain of</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasFilePath">http://kglids.org/ontology/data/hasFilePath</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasSemanticSimilarity">http://kglids.org/ontology/data/hasSemanticSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasTotalValueCount">http://kglids.org/ontology/data/hasTotalValueCount</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasContentSimilarity">http://kglids.org/ontology/data/hasContentSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasDeepPrimaryKeyForeignKeySimilarity">http://kglids.org/ontology/data/hasDeepPrimaryKeyForeignKeySimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasMissingValueCount">http://kglids.org/ontology/data/hasMissingValueCount</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasPrimaryKeyForeignKeySimilarity">http://kglids.org/ontology/data/hasPrimaryKeyForeignKeySimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasDistinctValueCount">http://kglids.org/ontology/data/hasDistinctValueCount</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasInclusionDependency">http://kglids.org/ontology/data/hasInclusionDependency</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasMedianValue">http://kglids.org/ontology/data/hasMedianValue</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasMaxValue">http://kglids.org/ontology/data/hasMaxValue</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasDeepEmbeddingContentSimilarity">http://kglids.org/ontology/data/hasDeepEmbeddingContentSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasDataType">http://kglids.org/ontology/data/hasDataType</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasColumnSimilarity">http://kglids.org/ontology/data/hasColumnSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasMinValue">http://kglids.org/ontology/data/hasMinValue</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasDeepEmbeddingContentSimilarity">http://kglids.org/ontology/data/hasDeepEmbeddingContentSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasInclusionDependency">http://kglids.org/ontology/data/hasInclusionDependency</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasContentSimilarity">http://kglids.org/ontology/data/hasContentSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasDeepPrimaryKeyForeignKeySimilarity">http://kglids.org/ontology/data/hasDeepPrimaryKeyForeignKeySimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasPrimaryKeyForeignKeySimilarity">http://kglids.org/ontology/data/hasPrimaryKeyForeignKeySimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/readsColumn">http://kglids.org/ontology/pipeline/readsColumn</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasColumnSimilarity">http://kglids.org/ontology/data/hasColumnSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasSemanticSimilarity">http://kglids.org/ontology/data/hasSemanticSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="DataItem">
        <h3>DataItem<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/DataItem</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/DataScienceItem">DataScienceItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>Sub-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/Dataset">Dataset</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/Table">Table</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/Source">Source</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/reads">http://kglids.org/ontology/pipeline/reads</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="DataScienceItem">
        <h3>DataScienceItem<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/DataScienceItem</code></td>
            </tr>
            <tr>
                <th>Sub-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/PipelineItem">PipelineItem</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/DataItem">DataItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In domain of</th>
                <td>
                    <a href="http://kglids.org/ontology/isPartOf">isPartOf</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/isPartOf">isPartOf</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Dataset">
        <h3>Dataset<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Dataset</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/DataItem">DataItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In domain of</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasDatasetSimilarity">http://kglids.org/ontology/data/hasDatasetSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasDatasetSimilarity">http://kglids.org/ontology/data/hasDatasetSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Function">
        <h3>Function<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Function</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/API">API</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsFunction">http://kglids.org/ontology/pipeline/callsFunction</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Library">
        <h3>Library<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Library</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/API">API</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsLibrary">http://kglids.org/ontology/pipeline/callsLibrary</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Package">
        <h3>Package<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Package</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/API">API</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsPackage">http://kglids.org/ontology/pipeline/callsPackage</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Pipeline">
        <h3>Pipeline<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Pipeline</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/PipelineItem">PipelineItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In domain of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/isWrittenOn">http://kglids.org/ontology/pipeline/isWrittenOn</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasSourceURL">http://kglids.org/ontology/pipeline/hasSourceURL</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasVotes">http://kglids.org/ontology/pipeline/hasVotes</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasScore">http://kglids.org/ontology/pipeline/hasScore</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasTag">http://kglids.org/ontology/pipeline/hasTag</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/isWrittenBy">http://kglids.org/ontology/pipeline/isWrittenBy</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="PipelineItem">
        <h3>PipelineItem<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/PipelineItem</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/DataScienceItem">DataScienceItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>Sub-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/Pipeline">Pipeline</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup><br/>
                    <a href="http://kglids.org/ontology/API">API</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Source">
        <h3>Source<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Source</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/DataItem">DataItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Statement">
        <h3>Statement<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Statement</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/PipelineItem">PipelineItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In domain of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/reads">http://kglids.org/ontology/pipeline/reads</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/callsPackage">http://kglids.org/ontology/pipeline/callsPackage</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasNextStatement">http://kglids.org/ontology/pipeline/hasNextStatement</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/readsTable">http://kglids.org/ontology/pipeline/readsTable</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasDataFlowTo">http://kglids.org/ontology/pipeline/hasDataFlowTo</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/readsColumn">http://kglids.org/ontology/pipeline/readsColumn</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/inControlFlow">http://kglids.org/ontology/pipeline/inControlFlow</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/callsLibrary">http://kglids.org/ontology/pipeline/callsLibrary</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/flowsTo">http://kglids.org/ontology/pipeline/flowsTo</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/callsFunction">http://kglids.org/ontology/pipeline/callsFunction</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/callsAPI">http://kglids.org/ontology/pipeline/callsAPI</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/callsClass">http://kglids.org/ontology/pipeline/callsClass</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasParameter">http://kglids.org/ontology/pipeline/hasParameter</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasText">http://kglids.org/ontology/pipeline/hasText</a><sup class="sup-dp" title="datatype property">dp</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/hasDataFlowTo">http://kglids.org/ontology/pipeline/hasDataFlowTo</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/flowsTo">http://kglids.org/ontology/pipeline/flowsTo</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/pipeline/hasNextStatement">http://kglids.org/ontology/pipeline/hasNextStatement</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity class" id="Table">
        <h3>Table<sup title="class" class="sup-c">c</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/Table</code></td>
            </tr>
            <tr>
                <th>Super-classes</th>
                <td>
                    <a href="http://kglids.org/ontology/DataItem">DataItem</a><sup class="sup-c" title="class">c</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In domain of</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasTableSimilarity">http://kglids.org/ontology/data/hasTableSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
            <tr>
                <th>In range of</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/readsTable">http://kglids.org/ontology/pipeline/readsTable</a><sup class="sup-op" title="object property">op</sup><br/>
                    <a href="http://kglids.org/ontology/data/hasTableSimilarity">http://kglids.org/ontology/data/hasTableSimilarity</a><sup class="sup-op" title="object property">op</sup><br/>
                </td>
            </tr>
        </table>
    </div>
</section>
<section id="objectproperties">
    <h4>Object Properties <span style="float:right; font-size:smaller;"><a href="">&uparrow;</a></span></h4>
    <ul class="hlist">
        <li><a href="#hasColumnSimilarity">hasColumnSimilarity</a></li>
        <li><a href="#hasContentSimilarity">hasContentSimilarity</a></li>
        <li><a href="#hasDatasetSimilarity">hasDatasetSimilarity</a></li>
        <li><a href="#hasDeepEmbeddingContentSimilarity">hasDeepEmbeddingContentSimilarity</a></li>
        <li><a href="#hasDeepPrimaryKeyForeignKeySimilarity">hasDeepPrimaryKeyForeignKeySimilarity</a></li>
        <li><a href="#hasInclusionDependency">hasInclusionDependency</a></li>
        <li><a href="#hasPrimaryKeyForeignKeySimilarity">hasPrimaryKeyForeignKeySimilarity</a></li>
        <li><a href="#hasSemanticSimilarity">hasSemanticSimilarity</a></li>
        <li><a href="#hasTableSimilarity">hasTableSimilarity</a></li>
        <li><a href="#isPartOf">isPartOf</a></li>
        <li><a href="#callsAPI">callsAPI</a></li>
        <li><a href="#callsClass">callsClass</a></li>
        <li><a href="#callsFunction">callsFunction</a></li>
        <li><a href="#callsLibrary">callsLibrary</a></li>
        <li><a href="#callsPackage">callsPackage</a></li>
        <li><a href="#flowsTo">flowsTo</a></li>
        <li><a href="#hasDataFlowTo">hasDataFlowTo</a></li>
        <li><a href="#hasNextStatement">hasNextStatement</a></li>
        <li><a href="#reads">reads</a></li>
        <li><a href="#readsColumn">readsColumn</a></li>
        <li><a href="#readsTable">readsTable</a></li>
    </ul>
    <div class="entity property" id="hasColumnSimilarity">
        <h3>hasColumnSimilarity<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasColumnSimilarity</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasContentSimilarity">
        <h3>hasContentSimilarity<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasContentSimilarity</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasColumnSimilarity">http://kglids.org/ontology/data/hasColumnSimilarity</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasDatasetSimilarity">
        <h3>hasDatasetSimilarity<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasDatasetSimilarity</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Dataset">Dataset</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Dataset">Dataset</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasDeepEmbeddingContentSimilarity">
        <h3>hasDeepEmbeddingContentSimilarity<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasDeepEmbeddingContentSimilarity</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasColumnSimilarity">http://kglids.org/ontology/data/hasColumnSimilarity</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasDeepPrimaryKeyForeignKeySimilarity">
        <h3>hasDeepPrimaryKeyForeignKeySimilarity<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasDeepPrimaryKeyForeignKeySimilarity</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasColumnSimilarity">http://kglids.org/ontology/data/hasColumnSimilarity</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasInclusionDependency">
        <h3>hasInclusionDependency<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasInclusionDependency</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasPrimaryKeyForeignKeySimilarity">
        <h3>hasPrimaryKeyForeignKeySimilarity<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasPrimaryKeyForeignKeySimilarity</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasColumnSimilarity">http://kglids.org/ontology/data/hasColumnSimilarity</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasSemanticSimilarity">
        <h3>hasSemanticSimilarity<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasSemanticSimilarity</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/data/hasColumnSimilarity">http://kglids.org/ontology/data/hasColumnSimilarity</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasTableSimilarity">
        <h3>hasTableSimilarity<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasTableSimilarity</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Table">Table</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Table">Table</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="isPartOf">
        <h3>isPartOf<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/isPartOf</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/DataScienceItem">DataScienceItem</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/DataScienceItem">DataScienceItem</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="callsAPI">
        <h3>callsAPI<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/callsAPI</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://www.w3.org/2002/07/owl#topObjectProperty">owl:topObjectProperty</a>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/API">API</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="callsClass">
        <h3>callsClass<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/callsClass</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsAPI">http://kglids.org/ontology/pipeline/callsAPI</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Class">Class</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="callsFunction">
        <h3>callsFunction<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/callsFunction</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsAPI">http://kglids.org/ontology/pipeline/callsAPI</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Function">Function</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="callsLibrary">
        <h3>callsLibrary<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/callsLibrary</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsAPI">http://kglids.org/ontology/pipeline/callsAPI</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Library">Library</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="callsPackage">
        <h3>callsPackage<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/callsPackage</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/callsAPI">http://kglids.org/ontology/pipeline/callsAPI</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Package">Package</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="flowsTo">
        <h3>flowsTo<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/flowsTo</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasDataFlowTo">
        <h3>hasDataFlowTo<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/hasDataFlowTo</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/flowsTo">http://kglids.org/ontology/pipeline/flowsTo</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasNextStatement">
        <h3>hasNextStatement<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/hasNextStatement</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/flowsTo">http://kglids.org/ontology/pipeline/flowsTo</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="reads">
        <h3>reads<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/reads</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/DataItem">DataItem</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="readsColumn">
        <h3>readsColumn<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/readsColumn</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/reads">http://kglids.org/ontology/pipeline/reads</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="readsTable">
        <h3>readsTable<sup title="object property" class="sup-op">op</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/readsTable</code></td>
            </tr>
            <tr>
                <th>Super-properties</th>
                <td>
                    <a href="http://kglids.org/ontology/pipeline/reads">http://kglids.org/ontology/pipeline/reads</a><sup class="sup-op" title="object property">op</sup>
                </td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Table">Table</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
</section>

<section id="datatypeproperties">
    <h4>Datatype Properties <span style="float:right; font-size:smaller;"><a href="">&uparrow;</a></span></h4>
    <ul class="hlist">
        <li><a href="#hasDataType">hasDataType</a></li>
        <li><a href="#hasDistinctValueCount">hasDistinctValueCount</a></li>
        <li><a href="#hasFilePath">hasFilePath</a></li>
        <li><a href="#hasMaxValue">hasMaxValue</a></li>
        <li><a href="#hasMedianValue">hasMedianValue</a></li>
        <li><a href="#hasMinValue">hasMinValue</a></li>
        <li><a href="#hasMissingValueCount">hasMissingValueCount</a></li>
        <li><a href="#hasTotalValueCount">hasTotalValueCount</a></li>
        <li><a href="#withCertainty">withCertainty</a></li>
        <li><a href="#hasParameter">hasParameter</a></li>
        <li><a href="#hasScore">hasScore</a></li>
        <li><a href="#hasSourceURL">hasSourceURL</a></li>
        <li><a href="#hasTag">hasTag</a></li>
        <li><a href="#hasText">hasText</a></li>
        <li><a href="#hasVotes">hasVotes</a></li>
        <li><a href="#inControlFlow">inControlFlow</a></li>
        <li><a href="#isWrittenBy">isWrittenBy</a></li>
        <li><a href="#isWrittenOn">isWrittenOn</a></li>
        <li><a href="#withParameterValue">withParameterValue</a></li>
    </ul>
    <div class="entity property" id="hasDataType">
        <h3>hasDataType<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasDataType</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasDistinctValueCount">
        <h3>hasDistinctValueCount<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasDistinctValueCount</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#int">xsd:int</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasFilePath">
        <h3>hasFilePath<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasFilePath</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasMaxValue">
        <h3>hasMaxValue<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasMaxValue</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#double">xsd:double</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasMedianValue">
        <h3>hasMedianValue<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasMedianValue</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#double">xsd:double</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasMinValue">
        <h3>hasMinValue<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasMinValue</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#double">xsd:double</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasMissingValueCount">
        <h3>hasMissingValueCount<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasMissingValueCount</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#int">xsd:int</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasTotalValueCount">
        <h3>hasTotalValueCount<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/hasTotalValueCount</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Column">Column</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#int">xsd:int</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="withCertainty">
        <h3>withCertainty<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/data/withCertainty</code></td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#double">xsd:double</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasParameter">
        <h3>hasParameter<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/hasParameter</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasScore">
        <h3>hasScore<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/hasScore</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Pipeline">Pipeline</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#double">xsd:double</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasSourceURL">
        <h3>hasSourceURL<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/hasSourceURL</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Pipeline">Pipeline</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasTag">
        <h3>hasTag<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/hasTag</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Pipeline">Pipeline</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasText">
        <h3>hasText<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/hasText</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="hasVotes">
        <h3>hasVotes<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/hasVotes</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Pipeline">Pipeline</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#int">xsd:int</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="inControlFlow">
        <h3>inControlFlow<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/inControlFlow</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Statement">Statement</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="isWrittenBy">
        <h3>isWrittenBy<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/isWrittenBy</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Pipeline">Pipeline</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="isWrittenOn">
        <h3>isWrittenOn<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/isWrittenOn</code></td>
            </tr>
            <tr>
                <th>Domain(s)</th>
                <td>
                    <a href="http://kglids.org/ontology/Pipeline">Pipeline</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
    <div class="entity property" id="withParameterValue">
        <h3>withParameterValue<sup title="datatype property" class="sup-dp">dp</sup></h3>
        <table>
            <tr>
                <th>URI</th>
                <td><code>http://kglids.org/ontology/pipeline/withParameterValue</code></td>
            </tr>
            <tr>
                <th>Range(s)</th>
                <td>
                    <a href="http://www.w3.org/2001/XMLSchema#string">xsd:string</a><sup class="sup-c" title="class">c</sup>
                </td>
            </tr>
        </table>
    </div>
</section>

  
  <section id="namedindividuals">
    <h4>Named Individuals <span style="float:right; font-size:smaller;"><a href="">&uparrow;</a></span></h4>
    <ul class="hlist">
    </ul>
</section>
  
  <section id="namespaces">
    <h4>Namespaces <span style="float:right; font-size:smaller;"><a href="">&uparrow;</a></span></h4>
    <dl>
        <dt>:</dt>
        <dd><code>http://kglids.org/ontology/</code></dd>
        <dt>owl</dt>
        <dd><code>http://www.w3.org/2002/07/owl#</code></dd>
        <dt>prov</dt>
        <dd><code>http://www.w3.org/ns/prov#</code></dd>
        <dt>rdf</dt>
        <dd><code>http://www.w3.org/1999/02/22-rdf-syntax-ns#</code></dd>
        <dt>rdfs</dt>
        <dd><code>http://www.w3.org/2000/01/rdf-schema#</code></dd>
        <dt>sdo</dt>
        <dd><code>https://schema.org/</code></dd>
        <dt>skos</dt>
        <dd><code>http://www.w3.org/2004/02/skos/core#</code></dd>
        <dt>xsd</dt>
        <dd><code>http://www.w3.org/2001/XMLSchema#</code></dd>
    </dl>
</section>
  <section id="legend">
      <h4>Legend</h4>
      <table class="entity">
          <tr><td><sup class="sup-c" title="Classes">c</sup></td><td>Classes</td></tr>
          <tr><td><sup class="sup-op" title="Object Properties">op</sup></td><td>Object Properties</td></tr>
          <tr><td><sup class="sup-fp" title="Functional Properties">fp</sup></td><td>Functional Properties</td></tr>
          <tr><td><sup class="sup-dp" title="Data Properties">dp</sup></td><td>Data Properties</td></tr>
          <tr><td><sup class="sup-ap" title="Annotation Properties">dp</sup></td><td>Annotation Properties</td></tr>
          <tr><td><sup class="sup-p" title="Properties">p</sup></td><td>Properties</td></tr>
          <tr><td><sup class="sup-ni" title="Named Individuals">ni</sup></td><td>Named Individuals</td></tr>
      </table>
  </section>

## Test run KGLiDS

<< link to colab file >>


## Technical Report



## Citing Our Work

If you find our work useful, please cite it in your research:



## Questions

For any questions please contact us at: mossad.helali@concordia.ca, essam.mansour@concordia.ca, shubham.vashisth@concordia.ca, philippe.carrier@concordia.ca
