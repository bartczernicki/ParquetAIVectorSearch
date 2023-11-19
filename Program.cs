using ParquetSharp;
using ParquetSharp.Arrow;
using System.IO;

namespace ParquetFilesPerformanceTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var parquet_files_directory = @"c:\users\bart\downloads\";
            var parquet_file_path_base = @"train-00025-of-00026-769064ea76815001.parquet";
            var parquet_file_path = Path.Combine(parquet_files_directory, parquet_file_path_base);

            Console.WriteLine("Parquet File Test");

            using (var parquetReader = new ParquetFileReader(parquet_file_path))
            {
                // Metadata
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
                        long groupNumRows = rowGroupReader.MetaData.NumRows;

                        var ids = rowGroupReader.Column(0).LogicalReader<string>().ReadAll(100);
                        var titles = rowGroupReader.Column(1).LogicalReader<string>().ReadAll(100);
                        var texts = rowGroupReader.Column(2).LogicalReader<string>().ReadAll(100);
                        var embeddings = rowGroupReader.Column(3).LogicalReader<double?[]>().ReadAll(100);
                    }
                }

                parquetReader.Close();
            }
        }
    }
}
