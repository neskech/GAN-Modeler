using g3;
namespace Main{

    class Processing{
        public static void Main(){
            const string in_path = "/Raw-Training/one.obj";
            const string out_path = "/Processed-Training/one.obj";
            const int new_vertex_count = 1000;
 
            DMesh3 mesh = StandardMeshReader.ReadMesh(in_path);
            Console.WriteLine("HEHHEHE" + mesh);
            Reducer r = new Reducer(mesh);
            r.ReduceToVertexCount(new_vertex_count);
            IOWriteResult result = StandardMeshWriter.WriteFile(out_path,
            new List<WriteMesh>() { new WriteMesh(mesh) }, WriteOptions.Defaults);
         
        }
    }
}