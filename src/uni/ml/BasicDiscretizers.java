package uni.ml;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class BasicDiscretizers {

	public Instances entropySplit(Instances inputTrain, int[] baseAttributes) throws Exception {
//		System.out.println("Entropy-based discretization");

		Instances outputTrain;

//		Distribution distribution = new Distribution(inputTrain);
//		EntropySplitCrit entropy = new EntropySplitCrit();
//		double result = entropy.splitCritValue(distribution);

		// This package uses Fayyad&Irani method which is entropy-based
		// discretization
		weka.filters.supervised.attribute.Discretize filter = new weka.filters.supervised.attribute.Discretize();
		if(baseAttributes != null)
			filter.setAttributeIndicesArray(baseAttributes);
		// filter.setOptions(new String[]{"-O"});
		filter.setInputFormat(inputTrain);
		outputTrain = Filter.useFilter(inputTrain, filter);

//		System.out.println(outputTrain);
		return outputTrain;
	}

	public Instances equalFrequencySplit(Instances inputTrain, int[] baseAttributes) throws Exception {
//		System.out.println("Equal-Frequency discretization");

		Instances outputTrain;

		Discretize filter = new Discretize();
		filter.setBins(10);
		filter.setUseEqualFrequency(true);
		if(baseAttributes != null)
			filter.setAttributeIndicesArray(baseAttributes);
		filter.setInputFormat(inputTrain);
		outputTrain = Filter.useFilter(inputTrain, filter);

//		System.out.println(outputTrain);
		return outputTrain;
	}

	public Instances equalWidthSplit(Instances inputTrain, int[] baseAttributes) throws Exception {
//		System.out.println("Equal-Width discretization");

		Instances outputTrain;

		Discretize filter = new Discretize();
		filter.setUseEqualFrequency(false);
		filter.setBins(10);
		
		if(baseAttributes != null){
			filter.setInvertSelection(true);
			filter.setAttributeIndicesArray(baseAttributes);
		}
		
		filter.setInputFormat(inputTrain);
		outputTrain = Filter.useFilter(inputTrain, filter);
//		outputTrain.setClassIndex(outputTrain.numAttributes() - 1);
//		Evaluation eval = new Evaluation(outputTrain);
		
//		System.out.println(eval.predictions());
//		ArffSaver saver = new ArffSaver();
//	    saver.setInstances(outputTrain);
//	    saver.setFile(new File("E:\\Projects\\JEE\\Eclipse\\University\\Result2.arff"));
//	    saver.writeBatch();
		

//		System.out.println(outputTrain);
		return outputTrain;
	}

	public BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public Instances loadData(BufferedReader dataFile) throws Exception {

		Instances result;
		result = new Instances(dataFile);
		result.setClassIndex(result.numAttributes() - 1);

		return result;
	}

}
