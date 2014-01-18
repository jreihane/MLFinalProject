package uni.ml;

import weka.core.Instances;

public class DiscretizedInstance {

	private Instances outputTrain = null;
	private double accuracy = 0.0;
	
	
	public Instances getOutputTrain() {
		return outputTrain;
	}
	public void setOutputTrain(Instances outputTrain) {
		this.outputTrain = outputTrain;
	}
	public double getAccuracy() {
		return accuracy;
	}
	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
	}
}
