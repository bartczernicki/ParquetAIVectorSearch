using ParquetSharp;
using ParquetSharp.Arrow;
using ParquetSharp.RowOriented;
using System.Collections.Concurrent;
using System.IO;
using System.Runtime.CompilerServices;

namespace ParquetFilesPerformanceTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Parquet File Test");

            var startTime = DateTime.Now;
            ConcurrentBag<DbPedia> dataSetDbPedias = new ConcurrentBag<DbPedia>();
            var recordCount = 0;

            // https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M 
            var parquet_files_directory = @"e:\data\dbpedia-entities-openai-1M\";
            var parquet_file_path_suffix = @"*.parquet";
            var parquet_files = Directory.GetFiles(parquet_files_directory, parquet_file_path_suffix);

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = (int) (Environment.ProcessorCount * 0.75)
            };

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
                            recordCount += groupNumRows;

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
                                    Embeddings = embeddings[i].Select(x => x).ToList()
                                };
                                dataSetDbPedias.Add(item);
                            }

                            Console.WriteLine($"File: {parquet_file} - Processed: {groupNumRows} rows.");
                        }
                    }

                    parquetReader.Close();
                }
            });

            // get elapsed time
            Console.WriteLine($"Time Taken: {(DateTime.Now - startTime).TotalSeconds} seconds");
            Console.WriteLine($"Total Records Processed: {recordCount}");
            Console.WriteLine($"Total Records in Concurrent Bag: {dataSetDbPedias.Count}");
        }

        //static void MergeColumns(string[] ids, string[] titles, string[] texts, double?[][]? embeddings)
        //{
        //    for (int i = 0; i < ids.Length; i++)
        //    {
        //        var item = new DbPedia
        //        {
        //            Id = ids[0],
        //            Title = titles[i],
        //            Text = texts[i],
        //            Embeddings = embeddings[i].Select(x => x).ToList()
        //        };

        //        dbPediaDataSet.Add(item);
        //    }
        //}
    }
}
