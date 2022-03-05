# KGLac Knowledge Graph Builder

The KG builder component is responsible for leveraging the profiles stored in the document database to build the KG. To do so, it uses RDF-star to represent the entities and their relationships. the KG builder uses the profiles to determine the different relationships between the entities. The relationships include semanticSimilarity, schemaSimilairty, primary key-secondary key (pkfk), and inclusion dependency. The generated graph will be then hosted on a RDF store supporting RDF-star (for now Apache Jena, Blazegraph)



### Components

------

#### SRC:

1. **api**
   1. <u>api.py:</u> Contains the different apis to be used in jupyter notebook.
2. **dataanalysis.py:** This folder was used to determine the semanticSimilarities before using the embedding. It should be removed.
3. **enums:**
   1. <u>relation.py:</u> An enum definining the various relationships reflected in the KG.
4. **out:** A folder containing the resultant KG
   1. <u>triples.ttl:</u> a file containing the tripes of the KG in a turtle serialization.
5. **storage:** A package containing the different source files dedicated to handle query the document DB and RDF store.
   1. <u>elasticsearch_client.py:</u> A source file containing the functions used to communicate with elasticsearch.
   2. <u>kglac_client.py:</u> A source file containing the functions used to communicate with the the RDF store. The function in this file are the implementation of function used in the <u>api.py</u> source file under **api** package.
   3. <u>kwtype.py:</u> An enu containing some metadata that was previously used. This should omitted.
   4. <u>query_templates:</u> A source file containing the query templates to used by the <u>kglac_client.py</u> functions to interact with the RDF store.
6. **word_embedding:** A package containing source files aimed to launch the word embedding server used for to determine the existence of the semanticSimilarity
   1. <u>embeddings_client:</u> A source file representing the client to the embedding server.
   2. <u>libclient.py:</u> A source file containing core functionalities used by the <u>embeddings_client</u> source file 
   3. <u>word_embeddings:</u> A source file containing the core functionalities offered by the server.
   4. <u>word_embeddings_services.py:</u> A source file containing the servcies offered by the embedding server. Here we also specify the path of the embeddings to load.
7. <u>config.py:</u> A source file containing parameters to run the KG builder.
8. <u>label.py:</u> A class to represent the object of the label predicate. The object in addition to the text of the label it should specify the language.
9. <u>rdf_builder.py:</u> A source file containing the different functions used to create the entities, determine the  different relationships mentioned above and dump the triples in the file <u>triple.ttl</u> found under **out**.
10. <u>rdf_builder_coordinator.py:</u> The main source file to run the fire the KG builder. It interacts with te various stores and the rdf_builder to create the KG.
11. <u>rdf_resource.py:</u> A class for an RDF triple component. It can be the subject, predicate, or the object. 
12. <u>triplet.py:</u> A class to represent the triples componsed. It uses recursion to model rdf-star.
13. <u>utils.py:</u> A source file containing helpful functions like generating the label which is used across different source files in different packages like <u>api.py</u> and <u>rdf_builder.py</u>. It also contains the code to generate the graph visualization using the graphviz library.

#### tests

​	This folder contains the different tests to run to make sure that the code works properly.



### How to run?

------

1. Connect to the vm and run elasticsearch:
   1. Open terminal and connect to the vm

   ```
   ssh -i path/to/ahmed-keypairs.pem ubuntu@206.12.92.210
   ```

   2. Go the folder of **app_servers**

   ```
   cd /mnt/discovery/app_servers
   ```

   3. Run ES 7.10.2

   ```
   elasticsearch-7.10.2/bin/elasticsearch
   ```

   Run ES 7.10.2**Note:** You can use kibana as a UI to see the profiles and the raw_data stored on the different indexes (profiles and raw_data respectively)

2. Run the the KG builder

   ```
   python rdf_builder_coordinator.py -opath out
   ```

3. Launch Blazegraph

   1. Open a new terminal and connect to the vm

   ```
   ssh -i path/to/ahmed-keypairs.pem  -L 9999:localhost:9999 ubuntu@206.12.92.210
   ```

   2. Run Blazegraph

   ```
   cd /mnt/discovery/app_servers/blazegraph
   java -server -Xmx4g -jar blazegraph.jar
   ```

   3. Create a namespace
      1. Open your browser
      2. Go to http://localhost:9999/blazegraph
      3. Go to namespaces.
      4. Create your NAMESPACE by specifying the name and set the mode to rdr to support rdf-star.
      5. Go to UPDATE. If you will upload the data specify the RDF DATA type and Turtle-RDR in the format. Elase speficy the path of <u>triples.ttl</u>
      6. Once the the data is loaded, go to WELCOME to start writing your query or using the APIs.