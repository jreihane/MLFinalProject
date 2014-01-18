package uni.ml;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instances;

public class AttributesOperations {

	public double findBestAttribute(Classifier base, Instances data) throws Exception{
		double result = 0;
		AttributeSelection attsel = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		// perform attribute selection
		attsel.SelectAttributes(data);
		int[] indices = attsel.selectedAttributes();
		for(int i = 0; i < indices.length; i++)
			System.out.println(indices[i]);
		return result;
	}
}
