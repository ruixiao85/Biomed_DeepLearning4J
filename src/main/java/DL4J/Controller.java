package DL4J;

import javafx.event.ActionEvent;
import javafx.scene.control.*;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.CheckBox;
import javafx.scene.input.MouseEvent;
import javafx.stage.DirectoryChooser;

import java.awt.Desktop;
import java.io.File;

public class Controller {
   public TextField tf_work_dir;
   public Button btn_set_dir;
   public TextField tf_train_dir, tf_pred_dir;
   public Label lbl_original_dir, lbl_target_dir;
   public TextField tf_original_dir, tf_target_dir;
   public Label lbl_epoch;
   public TextField tf_epoch;
   public Button btn_train, btn_train_predict, btn_predict;
   public Button btn_cancel;
   public ProgressBar progressBar;
   public Label lbl_status;
   public CheckBox chk_auto_all;

   public void initialize() {
      tf_work_dir.setText(System.getProperty("user.dir"));
//      chk_auto_all.setSelected(true);
   }

   public void setDir(ActionEvent actionEvent) {
      DirectoryChooser directoryChooser=new DirectoryChooser();
      directoryChooser.setTitle("Open Work Directory");
      directoryChooser.showDialog(btn_set_dir.getParent().getScene().getWindow());
   }
   public void trainDir(MouseEvent mouseEvent) {
      try {
         Desktop.getDesktop().open(new File(tf_work_dir.getText()+"//"+tf_train_dir.getText()));
      } catch (Exception ignored) { }
   }
   public void predDir(MouseEvent mouseEvent) {
      try {
         Desktop.getDesktop().open(new File(tf_work_dir.getText()+"//"+tf_pred_dir.getText()));
      } catch (Exception ignored) { }
   }
   public void setTarget(MouseEvent mouseEvent) { chk_auto_all.setSelected(false); }


}
