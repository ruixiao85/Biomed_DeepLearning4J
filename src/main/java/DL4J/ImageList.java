package DL4J;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import play.libs.F;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;


public class ImageList extends NativeImageLoader implements Serializable {

    public static final int height = 1040;
    public static final int width = 1392;
    public static final int channelin= 3;
    public static final int channelout= 2; // +[0,1] -[1,0]
    public static final String[] exts= new String[]{"jpg"};

    public ArrayList<String> ImageFiles;
    public ImageList(String trainDirectory, String nameSubfolder) {
        try {
            ImageFiles=new ArrayList<>();
            File base=new File(trainDirectory, nameSubfolder);
            List<File> imageFiles=(List<File>) FileUtils.listFiles(base, exts, true );
            for (File f : imageFiles) ImageFiles.add(base.toURI().relativize(f.toURI()).getPath());
        } catch (Exception ex) { throw new RuntimeException(ex); }
    }




}
