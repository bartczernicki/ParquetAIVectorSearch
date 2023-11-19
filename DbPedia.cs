using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParquetFilesPerformanceTest
{
    internal class DbPedia
    {
        public string Id { get; set; }
        public string Title { get; set; }
        public string Text { get; set; }
        public required List<double?> Embeddings { get; set; }
    }
}
