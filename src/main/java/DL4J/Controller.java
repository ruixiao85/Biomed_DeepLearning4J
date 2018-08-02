package DL4J;

import javafx.event.ActionEvent;
import javafx.scene.control.*;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.CheckBox;
import javafx.scene.input.MouseEvent;
import javafx.stage.DirectoryChooser;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.awt.Desktop;
import java.io.File;
import java.util.ArrayList;

import static DL4J.Core.train;

public class Controller {
   public TextField tf_work_dir;
   public Button btn_set_dir;
   public TextField tf_train_dir, tf_pred_dir;
   public Label lbl_original_dir, lbl_target_dir;
   public TextField tf_original_dir, tf_target_dir;
   public Label lbl_epoch; public TextField tf_epoch;
   public Label lbl_learning_rate; public TextField tf_learning_rate;
   public Button btn_train, btn_train_predict, btn_predict;
   public Button btn_cancel;
   public ProgressBar progressBar;
   public Label lbl_status;
   public CheckBox chk_auto_all;

   public void initialize()
   {
//      tf_work_dir.setText(System.getProperty("user.dir"));
      tf_work_dir.setText("D:\\Cel files\\2018-07.13 Adam Brenderia 2X LPS CGS");
      tf_train_dir.setText("071318 Cleaned 24H post cgs");
   }

   public void setDir(ActionEvent actionEvent) {
      DirectoryChooser directoryChooser=new DirectoryChooser();
      directoryChooser.setTitle("Open Work Directory");
      File dir=directoryChooser.showDialog(btn_set_dir.getParent().getScene().getWindow());
      tf_work_dir.setText(dir.getPath());
   }
   public void clickTrainDir(MouseEvent mouseEvent) {
      try {
         Desktop.getDesktop().open(new File(tf_work_dir.getText()+"//"+tf_train_dir.getText()));
      } catch (Exception ignored) { lbl_status.setText("unable to open train subfolder"); }
   }
   public void clickPredDir(MouseEvent mouseEvent) {
      try {
         Desktop.getDesktop().open(new File(tf_work_dir.getText()+"//"+tf_pred_dir.getText()));
      } catch (Exception ignored) { lbl_status.setText("unable to open pred subfolder"); }
   }
   public void setManualTargets(MouseEvent mouseEvent) { chk_auto_all.setSelected(false); }
   public UI registerUI(){
      UI ui=new UI();
      ui.setWorkDirectory(tf_work_dir.getText());
      ui.setTrainSubDirectory(tf_train_dir.getText());
      ui.setPredSubDirectory(tf_pred_dir.getText());
      ui.setNameOriginal(tf_original_dir.getText());
      ui.setNameTargets(tf_target_dir.getText());
      ui.setAutoAll(chk_auto_all.isSelected());
      ui.setNofEpoches(tf_epoch.getText());
      ui.setLearningRate(tf_learning_rate.getText());
      return ui;
   }
   public void startTrain(ActionEvent actionEvent) {
      UI ui=registerUI();
      train(ui);
   }
   public void startPredict(ActionEvent actionEvent) {
   }
   public void startTrainPred(ActionEvent actionEvent) {
      startTrain(actionEvent); startPredict(actionEvent);
   }

}
