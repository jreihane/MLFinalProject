package uni.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This Class is mainly used for applying operations on classes, like the first algorithm in article
 * @author Reihane Zekri
 *
 */
public class ClassifierOperations {
	
	private List<Attribute> attributes = null;
	
	public Map<String, Integer> getAttributesWithIndexes(Instances instances){
		Map<String, Integer> result = new HashMap<String, Integer>();
		List<Attribute> attributesList = new ArrayList<Attribute>();
		int i = 0;
		Enumeration<Attribute> attributes = instances.enumerateAttributes();
		
		while(attributes.hasMoreElements()){
			Attribute attr = attributes.nextElement();
			result.put(attr.name(), i);
			i++;
			
			attributesList.add(attr);
		}
		
		setAttributes(attributesList);
		
		return result;
	}
	
	// First Algorithm, Finds a non-parametric measure for dependence level of each attribute 
	public Map<Attribute, Double> getDepencencyLevel(Instances instances){
		Map<Attribute, Double> result = new HashMap<Attribute, Double>();
		
		Enumeration<Attribute> attributes = instances.enumerateAttributes();
		int numberOfInstances = instances.size();
		
		while(attributes.hasMoreElements()){
			Attribute attr = attributes.nextElement();
			
			// 1- Sort data for each attribute 
			instances.sort(attr);
			Map<Double,Integer> transitions = new HashMap<Double,Integer>();
			Map<Double,Integer> classInstances = new HashMap<Double,Integer>();
			double[] classValues = new double[numberOfInstances];
			
			// 2- for each element find its class
			for(int i = 0; i < numberOfInstances; i++){
				Instance currentInstance = instances.get(i);
				double classValue = currentInstance.classValue();
				classValues[i] = classValue;
				
				if(classInstances.get(classValue) != null){
					int numberOfInstancesInClass = classInstances.get(classValue);
					numberOfInstancesInClass++;
					classInstances.put(classValue, numberOfInstancesInClass);
				}
				else{
					classInstances.put(classValue, 1);
				}

				// 3- for each class find transitions
				if(i == 0){
					transitions.put(classValue, 0);
				}
				else if(i > 0 && classValue != classValues[i-1]){
					if(transitions.get(classValue) == null){
						transitions.put(classValue, 1);
					}
					else{
						int classValueTransition = transitions.get(classValue);
						classValueTransition++;
						transitions.put(classValue, classValueTransition);
					}
				}
				
			}
//			System.out.println(transitions);
			
			
			// ri = number of transitions
			// ni = number of instances with class value ci
			// pi = dependence level
			// n = number of instances
			// pj = sum((ri*ni)/((ni-1)*n))
			double pj = 0;
			for(int i = 0; i < numberOfInstances; i++){
				Instance currentInstance = instances.get(i);
				double classValue = currentInstance.classValue();

//				System.out.println(classValue + "   "  + transitions.get(classValue));
				
				int ni = classInstances.get(classValue);
				int ri = transitions.get(classValue);

				pj += (((double)ri/(double)(ni - 1)) * ((double)ni / (double)numberOfInstances));
//				System.out.println(pj);
			}
			
			result.put(attr, pj);
//			System.out.println(pj);
		}
		
//		System.out.println(result);
		return result;
	}
	
	public List<Attribute> sortDependencyLevels(Map<Attribute, Double> dependencyLevels){
		List<Attribute> result = new ArrayList<Attribute>();
		
		Map<Attribute, Double> mockDependency = new HashMap<Attribute, Double>();
		mockDependency.putAll(dependencyLevels);
		Set<Attribute> mockAttrs = mockDependency.keySet();
		Iterator<Attribute> it = mockAttrs.iterator();
		
		Collection<Double> values = dependencyLevels.values();
		List<Double> valuesList = new ArrayList<Double>();
		valuesList.addAll(values);
//		Double[] valArray = (Double[])values.toArray(); 
		Collections.sort(valuesList);
		
		while(it.hasNext()){
			Attribute currentAttr = it.next();
			for(int i = 0; i < valuesList.size(); i++){
				if(mockDependency.get(currentAttr) == valuesList.get(i)){
					result.add(currentAttr);
					mockDependency.remove(mockDependency.get(currentAttr));
				}
			}
//			result.add(dependencyLevels.get(valArray[i]));
		}
		
//		System.out.println(result);
		return result;
	}
	
	public double calculateAccuracy(List<NominalPrediction> predictions) {
        double correct = 0;
        
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = predictions.get(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }
        
        return 100 * correct / predictions.size();
    }
	
	public double findLargestPredictionAccuracy(Evaluation eval){
		double result = 0;
		
		return result;
	}

	public List<Attribute> getAttributes() {
		return attributes;
	}

	public void setAttributes(List<Attribute> attributes) {
		this.attributes = attributes;
	}
	
}
