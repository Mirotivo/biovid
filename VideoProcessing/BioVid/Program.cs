using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace BioVid
{
    class Program
    {
        static void Main(string[] args)
        {
            string rootpath = @"C:\Users\Amr Mostafa\Desktop\BioVid\videos_frontal";
            Array.ForEach(Directory.EnumerateFiles(rootpath, @"*.csv").ToArray(), (string path) => { File.Delete(path); });
            using (StreamWriter writer = new StreamWriter(@"C:\Users\Amr Mostafa\Desktop\BioVid\errors.txt"))
            {
                foreach (var item in Directory.EnumerateFiles(rootpath, @"*.mp4").OrderBy(itm => itm))
                {
                    try
                    {
                        using (Process proc = new Process())
                        {
                            string GeneratedFile = @"C:\Users\Amr Mostafa\OneDrive\Thesis\BioVid\VideoProcessing\Affectiva\bin\x64\Release\Data.csv";
                            if (File.Exists(GeneratedFile)) File.Delete(GeneratedFile);
                            ProcessStartInfo info = new ProcessStartInfo();
                            info.WorkingDirectory = @"C:\Users\Amr Mostafa\OneDrive\Thesis\BioVid\VideoProcessing\Affectiva\bin\x64\Release";// Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName);
                            info.FileName = "csharp-sample-app.exe";
                            info.Arguments = $@"-i ""{item}"" -d ""C:\Program Files\Affectiva\AffdexSDK\data""";
                            proc.StartInfo = info;
                            proc.OutputDataReceived += (object sender, DataReceivedEventArgs e) => { Console.WriteLine(e.Data); };
                            proc.ErrorDataReceived += (object sender, DataReceivedEventArgs e) => { Console.WriteLine(e.Data); };
                            proc.Start();
                            proc.WaitForExit();
                            GeneratedFile = Path.Combine(Path.GetDirectoryName(GeneratedFile), Path.ChangeExtension(Path.GetFileName(item), "csv"));
                            if (File.Exists(GeneratedFile)) File.Copy(GeneratedFile, Path.ChangeExtension(item, "csv"));
                        }
                    }
                    catch (Exception ex)
                    {
                        writer.WriteLine(item);
                    }
                }
            }
        }
    }
}
