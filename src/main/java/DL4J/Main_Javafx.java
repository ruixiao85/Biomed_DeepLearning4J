package DL4J;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

public class Main_Javafx extends Application
{
    public Parent root;
    @Override
    public void start(Stage primaryStage) throws Exception {
        root=FXMLLoader.load(getClass().getResource("UI.fxml"));
        primaryStage.setTitle("Biomed DL4J");
        primaryStage.setScene(new Scene(root, 600, 400));
        primaryStage.show();
    }
    public static void main(String[] args) throws Exception {


//            DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE); DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE); //DP4J double precision
//            DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT); DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT); //DP4J half precision to save memory

//            int cores = Runtime.getRuntime().availableProcessors();
//            System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", String.valueOf(16));
//            System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "5");

//            CudaEnvironment.getInstance().getConfiguration().setMaximumGridSize(512).setMaximumBlockSize(768); //allow larger grids/blocks
//            long gigCache=6L;
//            CudaEnvironment.getInstance().getConfiguration()
//                  .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
//                  .setMaximumDeviceCache(gigCache * 1024 * 1024 * 1024L)
//                  .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
//                  .setMaximumHostCache(gigCache * 1024 * 1024 * 1024L);//allow more cache

            launch(args);

    }
}
