package DL4J;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndexAll;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class ImagePair extends NativeImageLoader implements MultiDataSetIterator {
   private int batchSize = 1;
   private int index = 0;
   private int numExample = 0;
   private MultiDataSetPreProcessor preProcessor;
   private ArrayList<File> oriFiles, tarFiles;

   public ImagePair(String directory, String original, String target, ArrayList<String> comValid) {
      oriFiles=new ArrayList<>();
      tarFiles=new ArrayList<>();
      for (String f : comValid) {
         oriFiles.add(new File(directory+"/"+original, f));
         tarFiles.add(new File(directory+"/"+target, f));
      }
      numExample=comValid.size();

   }

   @Override
   public void reset() { index= 0; }
   @Override
   public boolean hasNext() { return index<numExample; }
   @Override
   public MultiDataSet next() { return next(batchSize); }
   @Override
   public MultiDataSet next(int items) {
      int end=index+items;
      if (end>=numExample) end=numExample-1;

      /*INDArray feature=Nd4j.zeros(new int[]{end-index, ImageList.channelin, ImageList.height, ImageList.width },'c');
      INDArray label=Nd4j.zeros(new int[]{end-index, ImageList.channelout, ImageList.height, ImageList.width},'c');
      for (int i=index; i<end; ++i){
         try {
            INDArray tf=asMatrix(oriFiles.get(i)).get(NDArrayIndex.point(0));
            tf.muli(1.0/255.);
//            System.out.println(tf.shapeInfoToString());
            feature.get(NDArrayIndex.point(i-index)).assign(tf);
            INDArray tc=asMatrix(tarFiles.get(i)).get(NDArrayIndex.point(0))
                  .get(NDArrayIndex.interval(0,1,true), NDArrayIndex.all(), NDArrayIndex.all());
            tc.muli(1.0/255.);
            System.out.println(tc.shapeInfoToString());
            tc.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())
                  .assign(tc.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).muli(-1).addi(1));
//            System.out.println(tc.shapeInfoToString());
            label.get(NDArrayIndex.point(i-index)).assign(tc);
         } catch (IOException e) {
            e.printStackTrace();
         }
      }
      MultiDataSet mds=new MultiDataSet(new INDArray[]{feature},new INDArray[]{label},null,null);
*/

      ArrayList<INDArray> feature=new ArrayList<>();
      ArrayList<INDArray> label=new ArrayList<>();
      for (int i=index; i<end; ++i){
         try {
            feature.add(asMatrix(oriFiles.get(i)).muli(1.0/255.0));
            label.add(asMatrix(tarFiles.get(i)).muli(1.0/255.0).get(NDArrayIndex.interval(0,0,true),NDArrayIndex.interval(2,2,true),NDArrayIndex.all(),NDArrayIndex.all()));
         } catch (IOException e) {
            e.printStackTrace();
         }
      }
      MultiDataSet mds=new MultiDataSet(new INDArray[]{Nd4j.vstack(feature)},new INDArray[]{Nd4j.vstack(label)});

      index=end;
      if (preProcessor != null) { preProcessor.preProcess(mds); }
      return mds;
   }

   @Override
   public void setPreProcessor(MultiDataSetPreProcessor preProcessor) { this.preProcessor = preProcessor; }
   @Override
   public MultiDataSetPreProcessor getPreProcessor() { return preProcessor; }

   @Override
   public boolean resetSupported() { return true; }
   @Override
   public boolean asyncSupported() { return false; }

}
