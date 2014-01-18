package uni.ml;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.w3c.dom.Attr;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;

public class Main {

	/**
	 * @param args
	 */
	@SuppressWarnings("unused")
	public static void main(String[] args) {
//		String fileName = "";
//		try {
//			BufferedReader bufferRead = new BufferedReader(new InputStreamReader(System.in));
//			fileName = bufferRead.readLine();
//		} catch (IOException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//			fileName = "iris";
//		}
		
		String[] fileNames = new String[]{"glass","haberman","ionosphere","iris","liver-disorders","hypothyroid","segment","vehicle"};
		double[] firstResults = new double[8];
		double[] finalResults = new double[8];
		
		for(int k = 0; k < fileNames.length; k++){
			BufferedReader dataFile = null;
			try {
				Instances inputTrain = null;
				 
				BasicMethods bestMethod = null;
				// NaiveBayes nB = new NaiveBayes();
	
				BasicDiscretizers basicDiscretizers = new BasicDiscretizers();
				ClassifierOperations classifierOps = new ClassifierOperations();
	
				dataFile = basicDiscretizers
						.readDataFile("C:\\Program Files\\Weka-3-7\\data\\new\\" + fileNames[k] + ".arff");
				inputTrain = basicDiscretizers.loadData(dataFile);
	
				// 1- rank continuous attributes by the nonparametric measure.
				Map<Attribute, Double> dependencyLevel = classifierOps
						.getDepencencyLevel(inputTrain);
				List<Attribute> sortedDependencies = classifierOps
						.sortDependencyLevels(dependencyLevel);
	
				// 2- for all continuous attributes,
				// find the best one that achieves the largest prediction accuracy
				// from the four basic discretization methods {
	
				// 2-1 Equal-width discretization
				double accuracy1 = applyDiscretization(BasicMethods.EQUAL_WIDTH,
						inputTrain, null).getAccuracy();
	//			System.out.println(accuracy1);
	
				// 2-2 Equal-frequency discretization
	//			System.out.println("=============================================");
	
				double accuracy2 = applyDiscretization(
						BasicMethods.EQUAL_FREQUENCY, inputTrain, null).getAccuracy();
	//			System.out.println(accuracy2);
	
				// 2-3 Entropy-based discretization
	//			System.out.println("=============================================");
				double accuracy3 = applyDiscretization(BasicMethods.ENTROPY_BASED,
						inputTrain, null).getAccuracy();
	//			System.out.println(accuracy3);
	
				// 2-4 find the highest accuracy among all accuracies
				double max = Math.max(accuracy1, accuracy2);
				max = Math.max(max, accuracy3);
				if (max == accuracy1)
					bestMethod = BasicMethods.EQUAL_WIDTH;
				else if (max == accuracy2)
					bestMethod = BasicMethods.EQUAL_FREQUENCY;
				else if (max == accuracy3)
					bestMethod = BasicMethods.ENTROPY_BASED;
				
				NaiveBayes nB1 = new NaiveBayes();
				nB1.buildClassifier(inputTrain);
				Evaluation eval2 = new Evaluation(inputTrain);
				eval2.crossValidateModel(nB1, inputTrain, 5, new Random(1));
//				System.out.println("first is: " + classifierOps.calculateAccuracy(eval2.predictions()));
				firstResults[k] = classifierOps.calculateAccuracy(eval2.predictions());
	
				// }// end of phase two
	
				
				Map<String, Integer> attributesIndex = classifierOps
						.getAttributesWithIndexes(inputTrain);
				Map<String, BasicMethods> bestMethod4Attributes = new HashMap<String, BasicMethods>();
				Instances outputTrain = inputTrain;
	//			System.out.println(attributesIndex);
	
				// 3- best basic discretization methods for continuous attributes
				// are determined one by one according to their ranks
				for (int i = 0; i < sortedDependencies.size(); i++) {
						// 3-1 test all basic methods for this attribute
						Attribute currentAttr = sortedDependencies.get(i);
	//
						BasicMethods bestMethodCurrent = findBestAccuracy(attributesIndex.get(currentAttr.name()),
																			currentAttr, outputTrain);
						
						int[] attributesIndecis0 = new int[] { attributesIndex
								.get(sortedDependencies.get(i).name()) };
						DiscretizedInstance disretizedResult0 = applyDiscretization(bestMethodCurrent,
								outputTrain, attributesIndecis0);
						outputTrain = disretizedResult0.getOutputTrain();
					// **********************************************************************
						
						bestMethod4Attributes.put(currentAttr.name(), bestMethodCurrent);
						
					// **********************************************************************
						
	
					// 3-2 test other attributes with best basic method
					
					for(int j = 0; j < i; j++){
						Attribute beforeCurrentAttr = sortedDependencies.get(j);
						int[] attributesIndecis = new int[] { attributesIndex
								.get(sortedDependencies.get(i).name()) };
						if(bestMethod4Attributes.get(beforeCurrentAttr.name()) != null){
							DiscretizedInstance disretizedResult = applyDiscretization(bestMethod4Attributes.get(beforeCurrentAttr.name()),
									outputTrain, attributesIndecis);
							
							outputTrain = disretizedResult.getOutputTrain();
						}
						else{
							DiscretizedInstance disretizedResult = applyDiscretization(bestMethod,
									outputTrain, attributesIndecis);
							
							outputTrain = disretizedResult.getOutputTrain();
						}
					}
					for(int j = i+1; j < sortedDependencies.size(); j++){
						Attribute afterCurrentAttr = sortedDependencies.get(j);
						int[] attributesIndecis = new int[] { attributesIndex
								.get(sortedDependencies.get(i).name()) };
						if(bestMethod4Attributes.get(afterCurrentAttr.name()) != null){
							DiscretizedInstance disretizedResult = applyDiscretization(bestMethod4Attributes.get(afterCurrentAttr),
									outputTrain, attributesIndecis);
							
							outputTrain = disretizedResult.getOutputTrain();
						}
						else{
							DiscretizedInstance disretizedResult = applyDiscretization(bestMethod,
									outputTrain, attributesIndecis);
							
							outputTrain = disretizedResult.getOutputTrain();
						}
					}
					
				}
				
				NaiveBayes nB = new NaiveBayes();
				nB.buildClassifier(outputTrain);
				Evaluation eval = new Evaluation(outputTrain);
				eval.crossValidateModel(nB, outputTrain, 5, new Random(1));
				
				finalResults[k] = classifierOps.calculateAccuracy(eval.predictions());
//				System.out.println("Final result is going to be printed:\n------------------------------------------");
//				System.out.println(classifierOps.calculateAccuracy(eval.predictions()));
//				System.out.println("------------------------------------------");
				
	
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} finally {
				try {
					dataFile.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		
//		System.out.println("Dataset:\t\t\tSame:\t\t\tHybrid result");
		System.out.printf("%15s  %15s   %15s %55s%n", "Dataset", "Same  ", "Hybrid result","");
		System.out.println("--------------------------------------------------------------");
//		System.out.printf("%1s  %-7s   %-7s   %-6s   %-6s%n", "n", "result1", "result2", "time1", "time2");
		for(int i = 0; i < fileNames.length; i++){
//			System.out.println(fileNames[i] + "\t\t" + firstResults[i] + "\t\t" + finalResults[i]);
			System.out.printf("%15s  %15f   %15f %55s%n", fileNames[i], firstResults[i], finalResults[i],"");
		}
	}
	
	public static BasicMethods findBestAccuracy(int attributesIndex, Attribute currentAttr, Instances inputTrain){
		BasicMethods result = null;
//		System.out.println(attributesIndex);
		int[] attributesIndecis = new int[] { attributesIndex };

		try {
			double attrAccuracy1 = applyDiscretization(
					BasicMethods.EQUAL_WIDTH, inputTrain,
					attributesIndecis).getAccuracy();
//			System.out.println(attrAccuracy1);

			double attrAccuracy2 = applyDiscretization(
					BasicMethods.EQUAL_FREQUENCY, inputTrain,
					attributesIndecis).getAccuracy();
//			System.out.println(attrAccuracy2);
			//
			double attrAccuracy3 = applyDiscretization(
					BasicMethods.ENTROPY_BASED, inputTrain,
					attributesIndecis).getAccuracy();
//			System.out.println(attrAccuracy3);
			
			double max = Math.max(attrAccuracy1, attrAccuracy2);
			max = Math.max(max, attrAccuracy3);
			if (max == attrAccuracy1)
				result = BasicMethods.EQUAL_WIDTH;
			else if (max == attrAccuracy2)
				result = BasicMethods.EQUAL_FREQUENCY;
			else if (max == attrAccuracy3)
				result = BasicMethods.ENTROPY_BASED;
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return result;
	}
	
	public static DiscretizedInstance applyDiscretization(BasicMethods basicMethod,
			Instances inputTrain, int[] attrIndices) throws Exception {
		
		DiscretizedInstance result = new DiscretizedInstance();

		BasicDiscretizers basicDiscretizers = new BasicDiscretizers();
		ClassifierOperations classifierOps = new ClassifierOperations();
		Instances outputTrain = null;
		NaiveBayes nB = new NaiveBayes();
		Evaluation eval = null;

		if (basicMethod == BasicMethods.EQUAL_WIDTH) {
			outputTrain = basicDiscretizers.equalWidthSplit(inputTrain,
					attrIndices);
		}

		else if (basicMethod == BasicMethods.EQUAL_FREQUENCY) {
			outputTrain = basicDiscretizers.equalFrequencySplit(inputTrain,
					attrIndices);
		} else if (basicMethod == BasicMethods.ENTROPY_BASED) {
			outputTrain = basicDiscretizers.entropySplit(inputTrain,
					attrIndices);
		}

		outputTrain.setClassIndex(outputTrain.numAttributes() - 1);
		result.setOutputTrain(outputTrain);
		
		nB.buildClassifier(outputTrain);
		eval = new Evaluation(outputTrain);
		eval.crossValidateModel(nB, outputTrain, 5, new Random(1));
		result.setAccuracy(classifierOps.calculateAccuracy(eval.predictions()));

		// System.out.println(eval.toSummaryString());
		// System.out.println(eval.toClassDetailsString());

		return result;
	}

}
