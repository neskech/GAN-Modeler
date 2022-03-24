using g3;
namespace Main{

    class Processing{
        public static void Main(String[] args){
            string[] files = new string[] { "one.obj", "two.obj", "three.obj" };

            bool useVert;
            if (args[0] == "v")
               useVert = true;
            else if (args[0] == "f")
                useVert = false;
            else
                throw new Exception("PUT IN A GOOD COMMAND LINE ARGUEMENT BRO WHAT ARE YOU DOING");

           int COUNT = Int32.Parse(args[1]);
            
            //TODO Make a new file called anayltics which takes the average vertex and face count across all the processed models
            for (int a = 0; a < files.Length; ++a)
            {
                string in_path = "Raw_Training/" + files[a];
                string out_path = "Processed_Training/" + files[a];


                DMesh3? mesh = StandardMeshReader.ReadMesh(in_path);
                Reducer r = new Reducer(mesh);

                if (useVert)
                     r.ReduceToVertexCount(COUNT);
                else
                     r.ReduceToTriangleCount(COUNT);

                Console.WriteLine(in_path + " || Num Verts: " + mesh.VertexCount);
                Console.WriteLine(in_path + " || Num Faces: " + mesh.TriangleCount);
                IOWriteResult result = StandardMeshWriter.WriteFile(out_path,
                new List<WriteMesh>() { new WriteMesh(mesh) }, WriteOptions.Defaults);
            }
          
         
        }
    }
}