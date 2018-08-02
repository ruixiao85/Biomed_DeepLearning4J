package DL4J;

public class UI {
   public UI(){
   }
   public UI(String wd, String td, String pd, String ori, String tar, int epoch, double lr){
      this.setWorkDirectory(wd);
      this.setTrainSubDirectory(td);
      this.setPredSubDirectory(pd);
      this.setNameOriginal(ori);
      this.setNameTargets(tar);
      this.setNofEpoches(epoch);
      this.setLearningRate(lr);
   }

   private String WorkDirectory;
   public String getWorkDirectory() { return WorkDirectory; }
   public void setWorkDirectory(String workDirectory) { WorkDirectory=workDirectory.replaceAll("\\\\","/"); }

   private String TrainSubDirectory;
   public String getTrainSubDirectory() { return TrainSubDirectory; }
   public void setTrainSubDirectory(String trainSubDirectory) { TrainSubDirectory=trainSubDirectory.trim(); }
   public String getTrainDirectory() { return WorkDirectory+"/"+TrainSubDirectory; }

   private String PredSubDirectory;
   public String getPredSubDirectory() { return PredSubDirectory; }
   public void setPredSubDirectory(String predSubDirectory) { PredSubDirectory=predSubDirectory.trim(); }
   public String getPredDirectory() { return WorkDirectory+"/"+PredSubDirectory; }

   private String NameOriginal;
   public String getNameOriginal() { return NameOriginal; }
   public void setNameOriginal(String nameOriginal) { NameOriginal=nameOriginal; }

   private String[] NameTargets;
   public String[] getNameTargets() { return NameTargets; }
   public void setNameTargets(String nameTargets) { NameTargets=nameTargets.split(","); }

   private boolean AutoAllTargets;
   public boolean getAutoAll() { return AutoAllTargets; }
   public void setAutoAll(boolean auto_all) { this.AutoAllTargets=auto_all; }

   private int NofEpoches;
   public int getNofEpoches() { return NofEpoches; }
   public void setNofEpoches(int nEpoch) { this.NofEpoches=nEpoch; }
   public void setNofEpoches(String nEpoch) { this.setNofEpoches(Integer.parseInt(nEpoch)); }

   private double LearningRate;
   public double getLearningRate() { return LearningRate; }
   public void setLearningRate(double learningRate) { this.LearningRate=learningRate; }
   public void setLearningRate(String learningRate) { this.setLearningRate(Double.valueOf(learningRate)); }

}
