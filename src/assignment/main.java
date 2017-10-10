package assignment;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Naive Bayes Classification
 * 
 * 
 * @author B00639414 - Larry Somers (Wolfstorm)
 */
public class main {
    
    final static String DATASET = System.getProperty("user.dir") +"/src/datasets/";
    
    //InputData - Reading in the data from the dataset
    public static Instances inputData(String FILENAME) throws Exception
    {
        //Get Dataset from Folder Location
        DataSource src = new DataSource(DATASET + FILENAME);
        Instances dataset = src.getDataSet();
        
        //Print to Console where the Gathered Dataset has come from
        System.out.println("Data Gathered from: " + DATASET + "" + FILENAME + ".");
        
        //Print Dataset Summary
        System.out.println(dataset.toSummaryString());
        return dataset;
    }//inputData
    
    //10 Crossfold Validation - Task D
    public static void ClassifierFolding(Instances dataset) throws Exception{
                
        //Load in Model
        NaiveBayes nb =  (NaiveBayes) SerializationHelper.read("B00639414-nb.model");
        
        //Set Class Index to Number of Atributes - 1
        if (dataset.classIndex() == -1)
            dataset.setClassIndex(dataset.numAttributes() - 1);
        
        //Print to Console the Following:
        // 1) Class Index Number - Always will be 5901
        // 2) Class Index Name - Always will be newsClass
        // 3) Number of Attributes - Always will be data.classIndex() + 1
        System.out.println("Class Index is: " + dataset.classIndex() + ".");
        System.out.println("Class Index Name: " + dataset.classAttribute().name() + ".");
        System.out.println("Number of Attributes is: " + dataset.numAttributes() + ".");
        
        //Create a New Evaluation Instance
        Evaluation eval = new Evaluation(dataset);
        
        //Evaluate the classifier
        eval.crossValidateModel(nb, dataset, 10, new Random(1));
        
        //Write Data to Model
        SerializationHelper.write("B00639414-nb.model", nb);
        
        //Print Results
        System.out.println("Printing Results to Cross Fold Evaluation - Task D");
        System.out.println("==================================================");
        printResults(eval);
    }//ClassifierFolding
    
    //Undertake 60/40% Split Partitioning of Data - Task E
    public static void percentagePartition(Instances dataset) throws Exception
    {
        //randomise the data
        dataset.randomize(new java.util.Random(0));

        //specify the training set to be 60%
        int trainSize = (int) Math.round(dataset.numInstances() * 0.6);

        //specify the testing set to be the remaining data
        int testSize = dataset.numInstances() - trainSize;

        //set up test and training data
        Instances trainData = new Instances(dataset, 0, trainSize);
        Instances testData = new Instances(dataset, trainSize, testSize);

        // train classifier
        Classifier BayesClass = new NaiveBayes();
        BayesClass.buildClassifier(trainData);

        //evaluate classifier
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(BayesClass, testData);
       
        //Print Results
        System.out.println("Printing Result: 60/40% Paritioning - Task E");
        System.out.println("============================================");
        printResults(eval);
    }//percentagePartition
    
    //Prints the results of the Evaluation to the Console Window
    public static void printResults(Evaluation eval) throws Exception 
    {
        //Print Evaluation Summary
        System.out.println(eval.toSummaryString());
        
        //Print Class Details
        System.out.println(eval.toClassDetailsString());
        
        //Print 20 x 20 Matrix
        System.out.println(eval.toMatrixString());
    }//Print Results
    
    public static void main(String[] args) throws Exception 
    {
        //Set Dataset Variable to the ARFF File
        Instances dataset = inputData("20newsgroups_StringToWordNEW.arff");
        
        //Call 10 Fold Validation Method - Task D
        System.out.println("Calling 10 Fold Validation Method");
        System.out.println("\n=================================");
        ClassifierFolding(dataset);
        
        //Call 60/40% Partitioning Method
        System.out.println("Calling 60/40% Partitioning Method");
        System.out.println("\n=================================");
        percentagePartition(dataset);
    }//main Method
    
}//Main Class
