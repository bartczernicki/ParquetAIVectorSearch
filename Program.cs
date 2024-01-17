using ParquetSharp;
using ParquetSharp.Arrow;
using ParquetSharp.RowOriented;
using System.Collections.Concurrent;
using System.IO;
using System.Runtime.CompilerServices;
using HNSW.Net;
using System.Collections.Frozen;
using System.Numerics.Tensors;
using System.Diagnostics;
using System.Runtime.Serialization.Formatters.Binary;
using System.Runtime.InteropServices;

namespace ParquetAIVectorSearch
{
    internal class Program
    {
        // Note this will not run on ARM processors

        private const string PARQUET_FILES_DIRECTORY = @"e:\data\dbpedia-entities-openai-1M\";
        private const string PARQUET_FILE_PATH_SUFFIX = @"*.parquet";
        private const int M_PARAMETER = 10; // determines the number of neighbors to consider when adding a new node to the graph



        static void Main(string[] args)
        {
            Console.WriteLine("Parquet File Test");

            var startTime = DateTime.Now;
            ConcurrentBag<DbPedia> dataSetDbPedias = new ConcurrentBag<DbPedia>();
            var recordCount = 0;

            // create name of file using the M_PARAMETER
            var GRAPH_FILE_NAME = $"graph_M{M_PARAMETER}.hnsw";

            // https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M 
            var parquet_files = Directory.GetFiles(PARQUET_FILES_DIRECTORY, PARQUET_FILE_PATH_SUFFIX);

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = (int) Math.Ceiling((Environment.ProcessorCount * 0.75))
            };

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
                                var item = new DbPedia
                                {
                                    Id = ids[i],
                                    Title = titles[i],
                                    Text = texts[i],
                                    Embeddings = embeddings[i].Select(x => (float) x).ToList()
                                };
                                dataSetDbPedias.Add(item);
                            }

                            Console.WriteLine($"File: {parquet_file} - Processed: {groupNumRows} rows.");
                        }
                    }

                    parquetReader.Close();
                }
            });

            // Parquet File Load - Get elapsed time & counts
            var endTimeOfParquetLoad = DateTime.Now;
            Console.WriteLine($"Time Taken: {(endTimeOfParquetLoad - startTime).TotalSeconds} seconds");
            Console.WriteLine($"Total Records Processed: {recordCount}");
            Console.WriteLine($"Total Records in Concurrent Bag: {dataSetDbPedias.Count}");


            Console.WriteLine($"Build ANN Graph using HNSW...");
            var NumVectors = dataSetDbPedias.Count;
            var batchSize = 10000;


            var hnswGraphParameters = new SmallWorld<float[], float>.Parameters()
            {
                M = M_PARAMETER,
                LevelLambda = 1 / Math.Log(M_PARAMETER), // should match M
                ExpandBestSelection = true,
                KeepPrunedConnections = true,
                EnableDistanceCacheForConstruction = true,
                NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic,
                InitialItemsSize = NumVectors,
                InitialDistanceCacheSize = NumVectors * 1024
        };

            var graph = new SmallWorld<float[], float>(DotProduct.DotProductOptimized, DefaultRandomGenerator.Instance,
                hnswGraphParameters, threadSafe: true);

            // enforce order as this is important for the graph to be built correctly
            var sampleVectors = dataSetDbPedias.OrderBy(a => a.Id).Select(x => x.Embeddings.ToArray()).ToList();

            var numberOfBatches = (int) Math.Ceiling((double) NumVectors / batchSize);

            for (int i = 0; i < numberOfBatches; i++)
            {
                var stopWatchBatch = Stopwatch.StartNew();
                stopWatchBatch.Start();

                var batch = sampleVectors.Skip(i * batchSize).Take(batchSize).ToList();
                graph.AddItems(batch);
                Console.WriteLine($"\nAdded {i + 1} of {numberOfBatches} \n");

                Console.WriteLine($"Time Taken for Batch {i+1}: {stopWatchBatch.ElapsedMilliseconds} ms.");
            }

            var endTimeOfGraphBuild = DateTime.Now;
            Console.WriteLine($"Time Taken to build ANN Graph: {(endTimeOfGraphBuild - endTimeOfParquetLoad).TotalSeconds} seconds");


            var searchVector = sampleVectors[0];// RandomVectors(1536, 1)[0];
            var results = graph.KNNSearch(searchVector, 20);

            var topMatches = results.OrderBy(a => a.Distance).Take(20);
            var endTimeOfSearch = DateTime.Now;
            Console.WriteLine($"Time Taken to search ANN Graph: {(endTimeOfSearch - endTimeOfGraphBuild).TotalSeconds} seconds");

            // Serialize the HNSW Graph & Persist to disk
            SaveHNSWGraph(graph, PARQUET_FILES_DIRECTORY, GRAPH_FILE_NAME);

            // De-Serialize the HNSW Graph & Persist to disk
            var loadedGraph = LoadHNSWGraph(PARQUET_FILES_DIRECTORY, GRAPH_FILE_NAME, sampleVectors);

            var loadedGraphResults = loadedGraph.KNNSearch(searchVector, 20);
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
                graph = SmallWorld<float[], float>.DeserializeGraph(vectors, DotProduct.DotProductOptimized, DefaultRandomGenerator.Instance, f, true);
            }

            stopWatch.Stop();
            Console.WriteLine($"Time Taken to load ANN (HNSW) Graph: {stopWatch.ElapsedMilliseconds} ms.");

            return graph;
        }
    }
}
