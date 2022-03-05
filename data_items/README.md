# Servers Installation

### ElasticSearch 7.10.2

1. Download ES 7.10.2

   ```
   wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
   ```

2. Get the SHA of the downloaded version

   ```
   wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
   ```

3. Compare SHA to the downloaded version

   ```
   shasum -a 512 -c elasticsearch-7.12.1-linux-x86_64.tar.gz.sha512
   ```

4. uncompress the downloaded folder

   ```
   tar -xzf elasticsearch-7.12.1-linux-x86_64.tar.gz
   ```

5. Start the server

   ```
   elasticsearch-7.10.2/bin/elasticsearch
   ```

6. Connect to localhost:9200



For more information, you check the main page: https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html

### Kibana

1. Download kibana 7.10.2

   ```
   curl -O https://artifacts.elastic.co/downloads/kibana/kibana-7.10.2-linux-x86_64.tar.gz
   ```

2. Get the SHA of the downloaded version and compare it to the downloaded version

   ```
   curl https://artifacts.elastic.co/downloads/kibana/kibana-7.10.2-linux-x86_64.tar.gz.sha512 | shasum -a 512 -c - 
   ```

3. uncompress the downloaded folder

   ```
   tar -xzf kibana-7.10.2-linux-x86_64.tar.gz
   ```

4. Start the server

   ```
   kibana-7.10.2-linux-x86_64/bin/kibana
   ```

   **Note** For kibana to run, you need to have already fired up ES 7.10.2

5. Connect to localhost:5601

**Note:** For more information check the page: https://www.elastic.co/guide/en/kibana/current/targz.html



### Blazegraph

1. Download Blazegraph

   ```
   wget https://github.com/blazegraph/database/releases/download/BLAZEGRAPH_2_1_6_RC/blazegraph.jar
   ```

2. Start Blazegraph

   ```
   java -server -Xmx4g -jar blazegraph.jar
   ```

3. Connect to Blazegraph

   ```
   localhost:9999
   ```

   **Note**: For more information check the main pages:

   - Download Page: https://github.com/blazegraph/database/releases/tag/BLAZEGRAPH_2_1_6_RC
   - Quick Start: https://github.com/blazegraph/database/wiki/Quick_Start 



### Apache Jena Fuseki

1. Download Apache Fuseki

   ```
   wget http://archive.apache.org/dist/jena/binaries/apache-jena-fuseki-3.16.0.tar.gz
   ```

2. Extract file from the tar folder

   ```
   tar -xzf apache-jena-fuseki-3.16.0.tar.gz
   ```

3. Run Jena Fuseki

   ```
   cd apache-jena-fuseki-3.16.0/
   ./fuseki-server
   ```

4. Connect to Apache Jena Fuseki

   ```
   localhost:3030
   ```

   ​