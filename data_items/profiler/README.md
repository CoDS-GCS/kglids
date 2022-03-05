## KGLAC Profiler

The KGLac profiler a scalable component responsible for parsing the semistructured data. It creates a data profile for each column stemming for the table. The content of the profile depends on the data type of the values of the column.

### Technologies

------

The profiler is **scalable**. It relies on multi-threading and Apache Spark in order to shard the tables into columns and profile each column.

#### Multi-threading

The profiler relies on threads to treat each table apart. The number of thread is configurable but the users needs to take into consideration the specifications of the machine being used. The reason behind using multi-threading is to allow KGLac profile to shard tables into columns to be profiled in parallel. In other words, each thread will handle a separate table. By handling, we mean shard the table into columns, determine the data type of each column, and profile them (collect some statistics and create column embeddings). 

#### Apache Spark

Apache Spark is a framework used for large data processing. Since we are dealing we data stemming from data lakes, Apache Spark is the best tool suited to handle processing them. Apache Spark is used to determine the data type of the column using its built-in functionalities when the file is loaded into a Spark DataFrame (more details below). In addition, Apache Spark allows collecting statistics about the columns loaded in the same DataFrame.

#### ElasticSearch

KGLac profiler processes the table to create profile for all the columns. These columns will be stored on ElasticSearch, a document database. In addition, the values for each column will also be stored on a ElasticSearch on a different index than the one dedicated to the profiles.



### Components

------

Therer are three folders under the profiler

1. ##### Config

The folder contains config.yml

​         **config.yml** a yaml file containing information about the datasets to be processed.

The structure of the file is as follows:

```
datasets:
 - name: dataset_name 
   type: type of file (csv, json, ...)
   path: path to the folder containing the dataset_name. (the tables are under path/dataset_name/)
   origin: separator for data values, (',' for csv)
```

2. **src**

    A folder containing the different packages and the source files.

   **analysis**: A package containing the sources files dedicated to analyze the tables. By analyzing we mean interpreting (determining the column data-type and profiling creation)

   ​	**interpreter: ** A package responsible for determining the data type of the columns of the loaded table. It contains:

   ​				<u>interpreter.py</u>:  A source file responsible for determining the data type of each column. (either numerical or textual). These types are determined when loading the csv file into a Spark dataframe with passing infer_schema set to true as an argument. This will however, require Spark to internally go over the content.

   ​                                  

   ​	**profile_creator:** A package responsible for creating the profile for each column depending on its datatype. It contains:

   ​                               ***analysers:***  a package containing classes for analysers where each is dedicated for one of the data type the interpreter is capable of determining. It contains:

   ​					<u>i_analyser.py:</u> An interface for all the analysers per data-type

   ​					<u>numerical_analyser.py:</u> A source file used to collect statistics about the the numerical columns. The collection of the statistics depends on the built_in function of the Spark DF, summary(). However, some statistics like the number_of_missing_values and the number_of_distinct_values are calculated using  the Resilient Distributed Datasets (RDD) data structure. 

   ​					<u>textual_analyser.py:</u> A source file used to collect statistics and embedding about the textual columns. We use RDD to get the distinct and missing values of per column. In addition, we determine a minHash for each column with size 512. To compute the minhash, we use the library datasketch.

   ​				<u>profile_creator.py</u>: A source file that uses the analysers to create data profiles.                   

   ​	<u>utils.py</u>: utility functions used across the source files in the package.

   ​

   **data:** A package containing the data structures to be used in addition to the functionalities to parse the config.yml file. it contains:

   ​	**tables:** A package containing the classes of the data structures used to handle the tables extracted from the datasets.  For each table type (csv, json,..) there should exist a class dedicated to parsing the file.  It contains:

   ​		<u>i_table.py</u>: An interface for the classes used to handle a table based on its type.

   ​		<u>csv_table.py:</u>  A class responsible for storing the information of csv files extracted from the datasets mentionedin the config.yml file. It retains the information about the path, dataset name, table name, and origin.

   ​	**utils**: A package containing the different common functionalities used in the parent package. It contains:

   ​		<u>file_type.py:</u> An enum dedicated to specify the file types to be considered for parsing. 

   ​		<u>yaml_parser.py</u> A source file used to parse the config.yaml file.

    	<u>data_profile.py:</u> A class that encapsulates the profile to be stored on the document database.

   ​	<u>raw_data.py</u> A class that encapsulates the column values to be stored on the document database.

   **orchesteation**: A package containing functionalities coordinate the different components of the profiler. It contains:

   ​	<u>orchestrator.py:</u> A class responsible for firing up elasticsearch, extracting the tables from the datasets specified in the config.yml file, and passing them to the worker thread to be processed.

   ​	<u>utils.py:</u> A souce file containing common functionalities like extracting teh tables from the specified datasets and getting the types.

   ​	<u>worker.py:</u> A class that implements thread. Each worker is responsible of handling the table handed by the orchestrator. handling a table means, interpret the columns, profile, and then store them on the document database.



​	**storage**

​		<u>i_documentDB.py:</u> An interface providing the functionalities the supported Document database should support.	 

​		<u>elastcisearchDB.py:</u> An implementation of the i_documentDB using elastidsearch

​		<u>utils.py:</u> A source file containing functionalities to serialize the profiles to be stored on the document databse.



​	<u>main.py:</u> Used to run the profiler. You can specify the number of thread to run here by passing the number in the process_tables as argument. In addition, you need to specify the the path to the config.yml fiule in the create_tables function. By default, the file is under profiler/src/config/.

​	<u>utils.py:</u> Containing the function that generates an id for the column based on the dataset, table, and file names. 

3. **tests:** Contains tests about the different functionalities in the different packages under src/

### How to run ?

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

   **Note:** You can use kibana as a UI to see the profiles and the raw_data stored on the different indexes (profiles and raw_data respectively)

2. Create the <u>config.yml</u> file and specify its path in the <u>main.py</u> source file to parse the datasets.

3. Run the profiler

   1. Connect to the vm using another terminal

   ```
   ssh -i path/to/ahmed-keypairs.pem ubuntu@206.12.92.210
   ```

   2. Go to the data_discovery folder

   ```
   cd /mnt/discovery/data_discovery
   ```

   3. Activate the environment

   ```
   source venv/bin/activate
   ```

   4. Run the profiler

   ```
   cd profiler/src
   pythom main.py
   ```

   ​