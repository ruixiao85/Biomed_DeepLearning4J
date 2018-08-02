package DL4J;

import org.deeplearning4j.nn.graph.ComputationGraph;

import java.util.ArrayList;

public class Core {
    public static void train(UI ui){
        String original=ui.getNameOriginal();
        ImageList ori=new ImageList(ui.getTrainDirectory(), original);
        ArrayList<String> oriValid=ori.ImageFiles;
        for (String target : ui.getNameTargets()) {
            ImageList tar=new ImageList(ui.getTrainDirectory(), target);
            ArrayList<String> comValid = new ArrayList<>(oriValid);
            comValid.retainAll(tar.ImageFiles);
            ImagePair imagePair=new ImagePair(ui.getTrainDirectory(), original, target, comValid);
            ComputationGraph model=Model.getUnet();
            model.fit(imagePair);
        }
    }


}
