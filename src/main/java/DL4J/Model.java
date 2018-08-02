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
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossFMeasure;

import static DL4J.ImageList.*;

public class Model {
//   public static ComputationGraph getZooUnet() {
//      UNet unet=new UNet();
//      ComputationGraphConfiguration conf = graph.build();
//      ComputationGraph model = new ComputationGraph(conf);
//      model.init();
//      System.out.println(model.summary());
//      return model;
//   }

   public static ComputationGraph getUnet() {
      ComputationGraphConfiguration.GraphBuilder graph=(new NeuralNetConfiguration.Builder())
         .seed(12342)
         .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
         .updater(new AdaDelta())
         .weightInit(WeightInit.RELU)
         .dist(new TruncatedNormalDistribution(0.0D, 0.5D))
         .l2(5.0E-5D)
         .miniBatch(true)
         .cacheMode(CacheMode.NONE)
         .trainingWorkspaceMode(WorkspaceMode.NONE)
         .inferenceWorkspaceMode(WorkspaceMode.NONE)
         .graphBuilder()
         .addInputs("input").setInputTypes(InputType.convolutional(height, width, channelin))
         .setOutputs("output")
         .backprop(true).pretrain(false);

      addUnetLayerManual(graph);


      ComputationGraphConfiguration conf = graph.build();
      ComputationGraph model = new ComputationGraph(conf);
      model.init();
      System.out.println(model.summary());
      return model;
   }

   private static void addUnetLayerManual(ComputationGraphConfiguration.GraphBuilder graph) {
//      ConvolutionLayer.AlgoMode cudnnAlgoMode=ConvolutionLayer.AlgoMode.PREFER_FASTEST;
      ConvolutionLayer.AlgoMode cudnnAlgoMode=ConvolutionLayer.AlgoMode.NO_WORKSPACE;
//      short[] f=new short[]{64,128,256,512,1024};
//      short[] f=new short[]{64,96,128,198,256};
      short[] f=new short[]{64,64,96,96,128};
      graph.addLayer("conv1-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[0])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "input")
         .addLayer("conv1-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[0])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv1-1")
         .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
               .build(), "conv1-2")

         .addLayer("conv2-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[1])
              .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
              .activation(Activation.RELU).build(), "pool1")
         .addLayer("conv2-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[1])
              .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
              .activation(Activation.RELU).build(), "conv2-1")
              .addLayer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                      .build(), "conv2-2")

         .addLayer("conv3-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[2])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "pool2")
         .addLayer("conv3-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[2])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv3-1")
         .addLayer("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
               .build(), "conv3-2")
/*
         .addLayer("conv4-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[3])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "pool3")
         .addLayer("conv4-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[3])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv4-1")
         .addLayer("drop4", new DropoutLayer.Builder(0.5).build(), "conv4-2")
         .addLayer("pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
               .build(), "drop4")

         .addLayer("conv5-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[4])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "pool4")
         .addLayer("conv5-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[4])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "conv5-1")
         .addLayer("drop5", new DropoutLayer.Builder(0.5).build(), "conv5-2")

         .addLayer("up4-1", new Upsampling2D.Builder(2).build(), "drop5")
         .addLayer("up4-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(f[3])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "up4-1")
         .addVertex("merge4", new MergeVertex(), "drop4", "up4-2")
         .addLayer("deconv4-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[3])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "merge4")
         .addLayer("deconv4-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[3])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "deconv4-1")*/

         .addLayer("up3-1", new Upsampling2D.Builder(2).build(), "pool3")
         .addLayer("up3-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(f[2])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "up3-1")
         .addVertex("merge3", new MergeVertex(), "conv3-2", "up3-2")
         .addLayer("deconv3-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[2])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "merge3")
         .addLayer("deconv3-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[2])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "deconv3-1")

         .addLayer("up2-1", new Upsampling2D.Builder(2).build(), "deconv3-2")
         .addLayer("up2-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(f[1])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "up2-1")
         .addVertex("merge2", new MergeVertex(), "conv2-2", "up2-2")
         .addLayer("deconv2-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[1])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "merge2")
         .addLayer("deconv2-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[1])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "deconv2-1")

         .addLayer("up1-1", new Upsampling2D.Builder(2).build(), "deconv2-2")
         .addLayer("up1-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(f[0])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "up1-1")
         .addVertex("merge1", new MergeVertex(), "conv1-2", "up1-2")
         .addLayer("deconv1-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[0])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "merge1")
         .addLayer("deconv1-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(f[0])
               .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.RELU).build(), "deconv1-1")

         .addLayer("deconv0", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1)
               .convolutionMode(ConvolutionMode.Truncate).cudnnAlgoMode(cudnnAlgoMode)
               .activation(Activation.SIGMOID).build(), "deconv1-2")
         .addLayer("output", new CnnLossLayer.Builder(new LossFMeasure()).build(), "deconv0"); // new LossBinaryXENT()
   }
   private static void addUnetLayer(ComputationGraphConfiguration.GraphBuilder graph, int[] filters) {
      ConvolutionLayer.AlgoMode cudnnAlgoMode=ConvolutionLayer.AlgoMode.PREFER_FASTEST;
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

              .addLayer("up4-1", new Upsampling2D.Builder(2).build(), "drop5")
              .addLayer("up4-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(512)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "up4-1")
              .addVertex("merge4", new MergeVertex(), "drop4", "up4-2")
              .addLayer("deconv4-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "merge4")
              .addLayer("deconv4-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "deconv4-1")

              .addLayer("up3-1", new Upsampling2D.Builder(2).build(), "deconv4-2")
              .addLayer("up3-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(256)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "up3-1")
              .addVertex("merge3", new MergeVertex(), "conv3-2", "up3-2")
              .addLayer("deconv3-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "merge3")
              .addLayer("deconv3-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "deconv3-1")

              .addLayer("up2-1", new Upsampling2D.Builder(2).build(), "deconv3-2")
              .addLayer("up2-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(128)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "up2-1")
              .addVertex("merge2", new MergeVertex(), "conv2-2", "up2-2")
              .addLayer("deconv2-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "merge2")
              .addLayer("deconv2-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "deconv2-1")

              .addLayer("up1-1", new Upsampling2D.Builder(2).build(), "deconv2-2")
              .addLayer("up1-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(64)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "up1-1")
              .addVertex("merge1", new MergeVertex(), "conv1-2", "up1-2")
              .addLayer("deconv1-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "merge1")
              .addLayer("deconv1-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                      .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.RELU).build(), "deconv1-1")

              .addLayer("deconv0", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1)
                      .convolutionMode(ConvolutionMode.Truncate).cudnnAlgoMode(cudnnAlgoMode)
                      .activation(Activation.SIGMOID).build(), "deconv1-2")
              .addLayer("output", new CnnLossLayer.Builder(LossFunctions.LossFunction.MCXENT).build(), "deconv0")
      ;
   }
}
