using ParquetSharp;
using ParquetSharp.Arrow;
using ParquetSharp.RowOriented;
using System.IO;

namespace ParquetFilesPerformanceTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M 
            var parquet_files_directory = @"e:\data\dbpedia-entities-openai-1M\";
            var parquet_file_path_suffix = @"*.parquet";
            var parquet_files = Directory.GetFiles(parquet_files_directory, parquet_file_path_suffix);
            var parquet_file_path = parquet_files.FirstOrDefault();

            Console.WriteLine("Parquet File Test");

            Parallel.ForEach(parquet_files, parquet_file =>
            //foreach (var parquet_file in parquet_files)
            {
                using (var parquetReader = new ParquetFileReader(parquet_file))
                {
                    Console.WriteLine($"File: {parquet_file}");

                    // Read Metadata
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

                            var ids = rowGroupReader.Column(0).LogicalReader<string>().ReadAll(groupNumRows);
                            var titles = rowGroupReader.Column(1).LogicalReader<string>().ReadAll(groupNumRows);
                            var texts = rowGroupReader.Column(2).LogicalReader<string>().ReadAll(groupNumRows);
                            var embeddings = rowGroupReader.Column(3).LogicalReader<double?[]>().ReadAll(groupNumRows);

                            Console.WriteLine($"File: {parquet_file} - Processed: {groupNumRows} rows.");
                        }
                    }

                    parquetReader.Close();
                }
            });
        }
    }
}
