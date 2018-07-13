package DL4J;

public class UI {

   private String WorkDirectory;
   public String getWorkDirectory() { return WorkDirectory; }
   public void setWorkDirectory(String workDirectory) { WorkDirectory=workDirectory.replaceAll("\\\\","/"); }

   private String TrainSubDirectory;
   public String getTrainSubDirectory() { return TrainSubDirectory; }
   public void setTrainSubDirectory(String trainSubDirectory) { TrainSubDirectory=trainSubDirectory.trim(); }
   public String getTrainDirectory() { return WorkDirectory+"/"+TrainSubDirectory; }

   private String PredSubDirectory;
   public String getPredSubDirectory() { return PredSubDirectory; }
   public void setPredSubDirectory(String PredSubDirectory) { PredSubDirectory=PredSubDirectory.trim(); }
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
   public void setNofEpoches(String nEpoch) { this.NofEpoches=Integer.parseInt(nEpoch); }

   private double LearningRate;
   public double getLearningRate() { return LearningRate; }
   public void setLearningRate(String learningRate) { this.LearningRate=Double.valueOf(learningRate); }

}
