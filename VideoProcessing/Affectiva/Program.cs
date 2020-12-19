using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace csharp_sample_app
{
    class Program
    {
        static public string VideoFile;
        static public string SignalsFile;
        static Affdex.Frame LoadFrameFromFile(string fileName)
        {
            Bitmap bitmap = new Bitmap(fileName);

            // Lock the bitmap's bits.
            Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bmpData = bitmap.LockBits(rect, ImageLockMode.ReadWrite, bitmap.PixelFormat);

            // Get the address of the first line.
            IntPtr ptr = bmpData.Scan0;

            // Declare an array to hold the bytes of the bitmap. 
            int numBytes = bitmap.Width * bitmap.Height * 3;
            byte[] rgbValues = new byte[numBytes];

            int data_x = 0;
            int ptr_x = 0;
            int row_bytes = bitmap.Width * 3;

            // The bitmap requires bitmap data to be byte aligned.
            // http://stackoverflow.com/questions/20743134/converting-opencv-image-to-gdi-bitmap-doesnt-work-depends-on-image-size

            for (int y = 0; y < bitmap.Height; y++)
            {
                Marshal.Copy(ptr + ptr_x, rgbValues, data_x, row_bytes);//(pixels, data_x, ptr + ptr_x, row_bytes);
                data_x += row_bytes;
                ptr_x += bmpData.Stride;
            }

            bitmap.UnlockBits(bmpData);

            return new Affdex.Frame(bitmap.Width, bitmap.Height, rgbValues, Affdex.Frame.COLOR_FORMAT.BGR);
        }
        [STAThread]
        static void Main(string[] args)
        {
            SignalsFile = "Data.csv";
            try
            {
                CmdOptions options = new CmdOptions();
                if (CommandLine.Parser.Default.ParseArguments(args, options))
                {
                    Affdex.Detector detector = null;
                    List<string> imgExts = new List<string> { ".bmp", ".jpg", ".gif", ".png", ".jpe" };
                    List<string> vidExts = new List<string> { ".avi", ".mov", ".flv", ".webm", ".wmv", ".mp4" };
                    bool isCamera = (options.Input.ToLower() == "camera");
                    bool isImage = imgExts.Any<string>(s => (options.Input.Contains(s) || options.Input.Contains(s.ToUpper())));
                    bool isVideo = (!isImage && !isCamera);

                    if (isCamera)
                    {
                        System.Console.WriteLine("Trying to process a camera feed...");
                        detector = new Affdex.CameraDetector(0, 30, 30, (uint)options.numFaces, (Affdex.FaceDetectorMode)options.faceMode);

                    }
                    else if (isImage)
                    {
                        System.Console.WriteLine("Trying to process a bitmap image..." + options.Input.ToString());
                        detector = new Affdex.PhotoDetector((uint)options.numFaces, (Affdex.FaceDetectorMode)options.faceMode);
                    }
                    else if (isVideo)
                    {
                        System.Console.WriteLine("Trying to process a video file..." + options.Input.ToString());
                        VideoFile = options.Input.ToString();
                        SignalsFile = Path.GetFileNameWithoutExtension(VideoFile) + ".csv";
                        detector = new Affdex.VideoDetector(30, (uint)options.numFaces, (Affdex.FaceDetectorMode)options.faceMode);
                    }
                    else
                    {
                        System.Console.WriteLine("File-Type not supported.");
                    }


                    if (detector != null)
                    {
                        ProcessVideo videoForm = new ProcessVideo(detector);
                        detector.setClassifierPath(options.DataFolder);
                        detector.setDetectAllEmotions(true);
                        detector.setDetectAllExpressions(true);
                        detector.setDetectAllEmojis(true);
                        detector.setDetectAllAppearances(true);
                        detector.start();
                        System.Console.WriteLine("Face detector mode = " + detector.getFaceDetectorMode().ToString());
                        if (isVideo) ((Affdex.VideoDetector)detector).process(options.Input);
                        else if (isImage) ((Affdex.PhotoDetector)detector).process(LoadFrameFromFile(options.Input));
                        videoForm.ShowDialog();
                        detector.stop();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Exception: " + ex.Message);
            }
        }

    }
}
