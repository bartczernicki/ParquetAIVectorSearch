using ParquetSharp;
using ParquetSharp.Arrow;
using ParquetSharp.RowOriented;
using System.Collections.Concurrent;
using System.IO;
using System.Runtime.CompilerServices;
using HNSW.Net;
using System.Collections.Frozen;
using System.Numerics.Tensors;

namespace ParquetAIVectorSearch
{
    internal class Program
    {
        // Note this will not run on ARM processors

        private const string PARQUET_FILES_DIRECTORY = @"c:\data\dbpedia-entities-openai-1M\";
        private const string PARQUET_FILE_PATH_SUFFIX = @"*.parquet";

        static void Main(string[] args)
        {
            Console.WriteLine("Parquet File Test");

            var startTime = DateTime.Now;
            ConcurrentBag<DbPedia> dataSetDbPedias = new ConcurrentBag<DbPedia>();
            var recordCount = 0;

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
            var batchSize = 5000;

            var hnswGraphParameters = new SmallWorld<float[], float>.Parameters()
            {
                M = 15,
                LevelLambda = 1 / Math.Log(15)
            };

            var graph = new SmallWorld<float[], float>(CosineDistance.SIMD, DefaultRandomGenerator.Instance,
                hnswGraphParameters, threadSafe: true);
            var sampleVectors = dataSetDbPedias.Select(x => x.Embeddings.ToArray()).ToList();

            var numberOfBatches = (int) Math.Ceiling((double) NumVectors / batchSize);

            for (int i = 0; i < numberOfBatches; i++)
            {
                var batch = sampleVectors.Skip(i * batchSize).Take(batchSize).ToList();
                graph.AddItems(batch);
                Console.WriteLine($"\nAdded {i + 1} of numberOfBatches \n");
            }

            var endTimeOfGraphBuild = DateTime.Now;
            Console.WriteLine($"Time Taken to build ANN Graph: {(endTimeOfGraphBuild - endTimeOfParquetLoad).TotalSeconds} seconds");


            var searchVector = sampleVectors[0];// RandomVectors(1536, 1)[0];
            var results = graph.KNNSearch(searchVector, 20);

            //var results = new List<VectorScore>(NumVectors);
            //for (var i = 0; i != NumVectors; i++)
            //{
            //    ReadOnlySpan<float> singleVector = sampleVectors.Slice(i, 1)[0];
            //    var similarityScore = TensorPrimitives.Dot(searchVector, singleVector);

            //    results.Add(new VectorScore { VectorIndex = i, SimilarityScore = similarityScore });
            //}

            var topMatches = results.OrderBy(a => a.Distance).Take(20);
            var endTimeOfSearch = DateTime.Now;
            Console.WriteLine($"Time Taken to search ANN Graph: {(endTimeOfSearch - endTimeOfGraphBuild).TotalSeconds} seconds");
        }

        private static List<float[]> RandomVectors(int vectorSize, int vectorsCount)
        {
            var vectors = new List<float[]>();

            for (int i = 0; i < vectorsCount; i++)
            {
                var vector = new float[vectorSize];
                DefaultRandomGenerator.Instance.NextFloats(vector);
                VectorUtils.Normalize(vector);
                vectors.Add(vector);
            }

            return vectors;
        }
    }
}
