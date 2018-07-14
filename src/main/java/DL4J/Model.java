package DL4J;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.layers.*;

import static DL4J.ImageList.*;

public class Model {

   public static ComputationGraph getUnet() {
      ConvolutionLayer.AlgoMode cudnnAlgoMode=ConvolutionLayer.AlgoMode.PREFER_FASTEST;
      ComputationGraphConfiguration.GraphBuilder graph=(new NeuralNetConfiguration.Builder())
            .seed(12342)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new AdaDelta())
            .weightInit(WeightInit.RELU)
            .dist(new TruncatedNormalDistribution(0.0D, 0.5D))
            .l2(5.0E-5D)
            .miniBatch(true)
            .cacheMode(CacheMode.NONE)
            .trainingWorkspaceMode(WorkspaceMode.ENABLED)
            .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
            .graphBuilder();

      graph.addLayer("conv1-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "input")
         .addLayer("conv1-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv1-1")
         .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
               .build(), "conv1-2")

         .addLayer("conv2-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "pool1")
         .addLayer("conv2-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv2-1")
         .addLayer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
               .build(), "conv2-2")

         .addLayer("conv3-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "pool2")
         .addLayer("conv3-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv3-1")
         .addLayer("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
               .build(), "conv3-2")

         .addLayer("conv4-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "pool3")
         .addLayer("conv4-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv4-1")
         .addLayer("drop4", new DropoutLayer.Builder(0.5).build(), "conv4-2")
         .addLayer("pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
               .build(), "drop4")

         .addLayer("conv5-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "pool4")
         .addLayer("conv5-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv5-1")
         .addLayer("drop5", new DropoutLayer.Builder(0.5).build(), "conv5-2")

         // up6
         .addLayer("up6-1", new Upsampling2D.Builder(2).build(), "drop5")
         .addLayer("up6-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(512)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "up6-1")
         .addVertex("merge6", new MergeVertex(), "drop4", "up6-2")
         .addLayer("conv6-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "merge6")
         .addLayer("conv6-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv6-1")

         // up7
         .addLayer("up7-1", new Upsampling2D.Builder(2).build(), "conv6-2")
         .addLayer("up7-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(256)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "up7-1")
         .addVertex("merge7", new MergeVertex(), "conv3-2", "up7-2")
         .addLayer("conv7-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "merge7")
         .addLayer("conv7-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv7-1")

         // up8
         .addLayer("up8-1", new Upsampling2D.Builder(2).build(), "conv7-2")
         .addLayer("up8-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(128)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "up8-1")
         .addVertex("merge8", new MergeVertex(), "conv2-2", "up8-2")
         .addLayer("conv8-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "merge8")
         .addLayer("conv8-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv8-1")

         // up9
         .addLayer("up9-1", new Upsampling2D.Builder(2).build(), "conv8-2")
         .addLayer("up9-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(64)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "up9-1")
         .addVertex("merge9", new MergeVertex(), "conv1-2", "up9-2")
         .addLayer("conv9-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "merge9")
         .addLayer("conv9-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv9-1")

         .addLayer("conv9-3", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(2)
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv9-2")
         .addLayer("conv10", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1)
               .convolutionMode(ConvolutionMode.Truncate).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.SIGMOID).build(), "conv9-3")
         .addLayer("output", new CnnLossLayer.Builder(LossFunctions.LossFunction.MCXENT).build(), "conv10")

         .addInputs("input").setInputTypes(InputType.convolutional(height, width, channelin))
         .setOutputs("output")
         .backprop(true).pretrain(false);

      ComputationGraphConfiguration conf = graph.build();
      ComputationGraph model = new ComputationGraph(conf);
      model.init();
      System.out.println(model.summary());
      return model;
   }
}
