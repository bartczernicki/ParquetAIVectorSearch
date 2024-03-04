using HNSW.Net;
using ParquetSharp;
using SharpToken;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace ParquetAIVectorSearch
{
    internal class Program
    {
        // Note this will not run on ARM processors

        private const string PARQUET_FILES_DIRECTORY = @"e:\data\dbpedia-entities-openai-1M\";
        private const string PARQUET_FILE_PATH_SUFFIX = @"*.parquet";
        private const int M_PARAMETER = 10; // determines the number of neighbors to consider when adding a new node to the graph
        private const int BATCHSIZE = 10000;
        // create name of file using the M_PARAMETER
        private static string GRAPH_FILE_NAME = $"graph_M{M_PARAMETER}.hnsw";

        static void Main(string[] args)
        {
            Console.WriteLine("Parquet File Test");

            var startTime = DateTime.Now;

            ConcurrentBag<DbPedia> dataSetDbPedias = new ConcurrentBag<DbPedia>();
            var recordCount = 0;

            // https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M 
            // Embeddings are 1536 dimensional - Title + Text
            var parquet_files = Directory.GetFiles(PARQUET_FILES_DIRECTORY, PARQUET_FILE_PATH_SUFFIX);

            var parallelOptions = new ParallelOptions
            {
                // Configure to use 75% of the available processors
                MaxDegreeOfParallelism = (int) Math.Ceiling((Environment.ProcessorCount * 0.75))
            };

            // Get the encoding for text-embedding-ada-002 or v3, set once as it is an expensive constructor
            var cl100kBaseEncoding = GptEncoding.GetEncoding("cl100k_base");

            // Load Parquet Files in parallel
            Parallel.ForEach(parquet_files, parallelOptions, parquet_file =>
            {
                using (var parquetReader = new ParquetFileReader(parquet_file))
                {
                    Console.WriteLine($"File: {parquet_file}");

                    // Read Metadata
                    // This should be the same if you are reading the same type of parquet files
                    int numColumns = parquetReader.FileMetaData.NumColumns;
                    long numRows = parquetReader.FileMetaData.NumRows;
                    int numRowGroups = parquetReader.FileMetaData.NumRowGroups;
                    IReadOnlyDictionary<string, string> metadata = parquetReader.FileMetaData.KeyValueMetadata;

                    SchemaDescriptor schema = parquetReader.FileMetaData.Schema;
                    for (int columnIndex = 0; columnIndex < schema.NumColumns; ++columnIndex)
                    {
                        ColumnDescriptor column = schema.Column(columnIndex);
                        string columnName = column.Name;
                        var dataType = column.LogicalType;
                        string dataTypeString = dataType.ToString();
                    }

                    for (int rowGroup = 0; rowGroup < parquetReader.FileMetaData.NumRowGroups; ++rowGroup)
                    {
                        using (var rowGroupReader = parquetReader.RowGroup(rowGroup))
                        {
                            var groupNumRows = (int)rowGroupReader.MetaData.NumRows;
                            recordCount += groupNumRows; //Used to match processed and records inserted into ConcurrentBag

                            var ids = rowGroupReader.Column(0).LogicalReader<string>().ReadAll(groupNumRows);
                            var titles = rowGroupReader.Column(1).LogicalReader<string>().ReadAll(groupNumRows);
                            var texts = rowGroupReader.Column(2).LogicalReader<string>().ReadAll(groupNumRows);
                            var embeddings = rowGroupReader.Column(3).LogicalReader<double?[]>().ReadAll(groupNumRows);

                            for (int i = 0; i < ids.Length; i++)
                            {
                                var titleAndText = titles[i] + ' ' + texts[i];
                                var encodedTokens = cl100kBaseEncoding.Encode(titleAndText);

                                var item = new DbPedia
                                {
                                    Id = ids[i],
                                    Title = titles[i],
                                    Text = texts[i],
                                    Embeddings = embeddings[i].Select(x => (float) x).ToList(),
                                    TokenCount = encodedTokens.Count
                                };
                                dataSetDbPedias.Add(item);
                            }

                            Console.WriteLine($"File: {parquet_file} - Processed: {groupNumRows} rows.");
                        }
                    }

                    parquetReader.Close();
                }
            });

            // Write out the first 1000 records to a json file
            var json = System.Text.Json.JsonSerializer.Serialize(dataSetDbPedias.Take(100));
            File.WriteAllText(Path.Combine(PARQUET_FILES_DIRECTORY, "dbpedias.json"), json);

            // Enforce order as this is important for the graph to be built correctly
            var dataSetDbPediasOrdered = dataSetDbPedias.OrderBy(a => a.Id).ToList();
            var sampleVectors = dataSetDbPediasOrdered.Select(x => x.Embeddings.ToArray()).ToList();

            // Parquet File Load - Get elapsed time & counts
            var endTimeOfParquetLoad = DateTime.Now;
            var totalTokens = dataSetDbPedias.Sum(x => x.TokenCount);
            var totalTokensStringWithCommas = totalTokens.ToString("N0");
            var priceToEncodeAdav2Token = 0.1/1000000; // $0.10 per 1M tokens
            var priceToEncodeAdav3TokenSmall = 0.02/1000000; // $0.02 per 1M tokens
            var tokensCostAdav2InDollarsString = (totalTokens * priceToEncodeAdav2Token).ToString("C");
            var tokensCostAdav3InDollarsString = (totalTokens * priceToEncodeAdav3TokenSmall).ToString("C");

            Console.WriteLine($"Time Taken: {(endTimeOfParquetLoad - startTime).TotalSeconds} seconds");
            Console.WriteLine($"Total Records Processed: {recordCount.ToString("N0")}");
            Console.WriteLine($"Total Records in Concurrent Bag: {dataSetDbPedias.Count.ToString("N0")}");
            Console.WriteLine($"Total Tokens (encoded) in Concurrent Bag: {totalTokensStringWithCommas}");
            Console.WriteLine($"Total Cost to process tokens (v2) in Concurrent Bag: {tokensCostAdav2InDollarsString}");
            Console.WriteLine($"Total Cost to process tokens (v3) in Concurrent Bag: {tokensCostAdav3InDollarsString}");

            Console.WriteLine($"Build ANN Graph using HNSW...");
            var NumVectors = dataSetDbPedias.Count;

            var hnswGraphParameters = new SmallWorld<float[], float>.Parameters()
            {
                M = M_PARAMETER,
                LevelLambda = 1 / Math.Log(M_PARAMETER), // should match M
                ExpandBestSelection = true,
                KeepPrunedConnections = true,
                EnableDistanceCacheForConstruction = true,
                NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic,
                InitialItemsSize = 100000,
                InitialDistanceCacheSize = 100000 * 1024
            };

            //// Option 1 - Process in parallel all of the batches to a single graph
            //var graphs = new List<SmallWorld<float[], float>>();

            //for (int i = 0; i != 10; i++)
            //{
            //    var graphLocal = new SmallWorld<float[], float>(DotProductDistance.DotProductOptimized, DefaultRandomGenerator.Instance,
            //        hnswGraphParameters, threadSafe: true);

            //    graphs.Add(graphLocal);
            //}   

            //var graph = new SmallWorld<float[], float>(DotProductDistance.DotProductOptimized, DefaultRandomGenerator.Instance,
            //    hnswGraphParameters, threadSafe: true);

            //// Enforce order as this is important for the graph to be built correctly
            //var dataSetDbPediasOrdered = dataSetDbPedias.OrderBy(a => a.Id).ToList();
            //var sampleVectors = dataSetDbPediasOrdered.Select(x => x.Embeddings.ToArray()).ToList();

            //// Chunk the sampleVector into 10 equal batches
            //var dbPediasOrderedPartitions = dataSetDbPediasOrdered.Chunk(sampleVectors.Count / 10).ToList();

            //var numberOfBatches = (int) Math.Ceiling((double) NumVectors / BATCHSIZE);

            //Parallel.For(0, dbPediasOrderedPartitions.Count, parallelOptions, i =>
            //{
            //    var partition = dbPediasOrderedPartitions[i].Select(x => x.Embeddings.ToArray()).ToList();
            //    var numberOfBatches = (int) Math.Ceiling((double)partition.Count / BATCHSIZE);

            //    for (int j = 0; j < numberOfBatches; j++)
            //    {
            //        var partitionBatch = partition.Skip(j * BATCHSIZE).Take(BATCHSIZE).ToList();
            //        graphs[i].AddItems(partitionBatch);
            //        Console.WriteLine($"Vector Partition: {i+1}, Added {j + 1} of {numberOfBatches} \n");
            //    }
            //});


            // Option 2 - Process Sequentially all of the batches to a single graph
            //var numberOfBatches = (int)Math.Ceiling((double)NumVectors / BATCHSIZE);

            //var graph = new SmallWorld<float[], float>(DotProductDistance.DotProductOptimized, DefaultRandomGenerator.Instance,
            //    hnswGraphParameters, threadSafe: true);

            //for (int i = 0; i < numberOfBatches; i++)
            //{
            //    var stopWatchBatch = Stopwatch.StartNew();
            //    stopWatchBatch.Start();

            //    var batch = sampleVectors.Skip(i * BATCHSIZE).Take(BATCHSIZE).ToList();
            //    graph.AddItems(batch);
            //    Console.WriteLine($"\nAdded {i + 1} of {numberOfBatches} \n");

            //    Console.WriteLine($"Time Taken for Batch {i + 1}: {stopWatchBatch.ElapsedMilliseconds} ms.");
            //}
            //// Serialize the HNSW Graph & Persist to disk
            //SaveHNSWGraph(graph, PARQUET_FILES_DIRECTORY, GRAPH_FILE_NAME);

            //var endTimeOfGraphBuild = DateTime.Now;
            //Console.WriteLine($"Time Taken to build ANN Graph: {(endTimeOfGraphBuild - endTimeOfParquetLoad).TotalSeconds} seconds");


            //for (int i = 0; i != 10; i++)
            //{
            //    SaveHNSWGraph(graphs[i], PARQUET_FILES_DIRECTORY, $"graph_M{M_PARAMETER}_partition{i + 1}.hnsw");
            //}


            // SEARCH

            var searchVector = sampleVectors[0];// RandomVectors(1536, 1)[0];
            
            //// Search the ANN Graph (in-memory)
            // var results = graph.KNNSearch(searchVector, 20);
            //var topMatches = results.OrderBy(a => a.Distance).Take(20);

            // De-Serialize the HNSW Graph & Persist to disk
            var loadedGraph = LoadHNSWGraph(PARQUET_FILES_DIRECTORY, GRAPH_FILE_NAME, sampleVectors);

            var loadedGraphResults = loadedGraph.KNNSearch(searchVector, 20);
            var topMatchesLoaded = loadedGraphResults.OrderBy(a => a.Distance).Take(20);

            var endTimeOfSearch = DateTime.Now;

            //var topMatchesDistanceSum = topMatches.Sum(x => x.Distance);
            var topMatchesLoadedDistanceSum = topMatchesLoaded.Sum(x => x.Distance);

            //Console.WriteLine($"Top Matches Distance: {0}", topMatchesDistanceSum);
            Console.WriteLine($"Top Matches Loaded Distance: {topMatchesLoadedDistanceSum}");
            //Console.WriteLine($"Time Taken to search ANN Graph: {(endTimeOfSearch - endTimeOfGraphBuild).TotalSeconds} seconds");
        }

        private static void SaveHNSWGraph(SmallWorld<float[], float> world, string directoryName, string fileName)
        {
            var filePath = Path.Combine(directoryName, fileName);
            Console.WriteLine($"Saving HNSW graph to '{filePath}'... ");
            var startTimeOfSave = DateTime.Now;

            // Save the graph to disk (doesn't save the vector data)
            using (var f = File.Open(filePath, FileMode.Create))
            {
                world.SerializeGraph(f);
            }

            Console.WriteLine($"Time Taken to save ANN (HNSW) Graph: {(DateTime.Now - startTimeOfSave).TotalMilliseconds} ms.");
        }

        private static SmallWorld<float[], float> LoadHNSWGraph(string directoryName, string fileName, List<float[]> vectors)
        {
            var stopWatch = Stopwatch.StartNew();
            stopWatch.Start();
            var filePath = Path.Combine(directoryName, fileName);

            SmallWorld<float[], float> graph;

            using (var f = File.OpenRead(filePath))
            {
                graph = SmallWorld<float[], float>.DeserializeGraph(vectors, DotProductDistance.DotProductOptimized, DefaultRandomGenerator.Instance, f, true);
            }

            stopWatch.Stop();
            Console.WriteLine($"Time Taken to load ANN (HNSW) Graph: {stopWatch.ElapsedMilliseconds} ms.");

            return graph;
        }
    }
}
