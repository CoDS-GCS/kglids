## Data Profiling


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



​	<u>main.py:</u> Used to run the profiler. You can specify the number of thread to run here by passing the number in the process_tables as argument. In addition, you need to specify the the path to the config.yml fiule in the create_tables function. By default, the file is under profiler/src/config/.

​	<u>utils.py:</u> Containing the function that generates an id for the column based on the dataset, table, and file names. 

3. **tests:** Contains tests about the different functionalities in the different packages under src/
