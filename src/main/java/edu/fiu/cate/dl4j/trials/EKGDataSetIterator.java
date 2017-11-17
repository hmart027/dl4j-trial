package edu.fiu.cate.dl4j.trials;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import edu.fiu.cate.wfdb.HeaderFile;

public class EKGDataSetIterator implements DataSetIterator{
	private static final long serialVersionUID = 3904541407259864248L;
	
	private HeaderFile head = null;
	private double[][] data = null; // index0: channel; index1: sample
	private int batchSize;		// number of training samples in a batch
	private int exampleLength; 	// length of the training samples in sample count
	private int featureCount;  	// number of features in the signal (ie. channels)
	private int sampleCount;   	// number of samples in the file
	private int numberOfLabels=1; // number of distinct labels 
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

	private EKGDataSetIterator(){}
	
	private void init(int batchSize, int exampleLength){
		this.data = head.loadSignal();
		this.batchSize = batchSize;
		this.exampleLength = exampleLength;
		this.featureCount = data.length;
		this.sampleCount = data[0].length;
		initializeOffsets();
	}
	
	private void initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (sampleCount - 1) / exampleLength - 2;   //-2: for end index, and for partial example
        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * exampleLength);
        }
        Random rng = new Random();
        Collections.shuffle(exampleStartOffsets, rng);
    }
	
	public static EKGDataSetIterator Build(String headerPath, int batchSize, int exampleLength) throws FileNotFoundException{
		EKGDataSetIterator itter = new EKGDataSetIterator();
		File file = new File(headerPath);
		itter.head = HeaderFile.load(file.getParent(), file.getName());
		if(itter.head == null)
			throw new FileNotFoundException(headerPath+": does not exist");
		itter.init(batchSize, exampleLength);
		return itter;
	}
	
	public void setLabels(){
		
	}
	
	@Override
	public boolean hasNext() {
		return exampleStartOffsets.size() > 0;
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

	@Override
	public DataSet next(int size) {
		if( exampleStartOffsets.size() == 0 ) throw new NoSuchElementException();
        int currMinibatchSize = Math.min(size, exampleStartOffsets.size());
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of input features)
        // dimension 2 = length of each time series/example
		INDArray input = Nd4j.create(new int[]{currMinibatchSize,featureCount,exampleLength}, 'f');
		INDArray labels = Nd4j.create(new int[]{currMinibatchSize,numberOfLabels,exampleLength}, 'f');
		for( int i=0; i<currMinibatchSize; i++ ){
            int t0 = exampleStartOffsets.removeFirst();
            int tN = t0 + exampleLength;
            for( int j=t0+1; j<tN; j++){
            	for(int s=0; s<featureCount; s++){
            		input.putScalar(new int[]{i,s,j}, data[s][j]);
            	}
                labels.putScalar(new int[]{i,1,j}, 1.0);
            }
        }
		return new DataSet(input,labels);
	}
	
	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return totalExamples() - exampleStartOffsets.size();
	}

	@Override
	public List<String> getLabels() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public int totalOutcomes() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int inputColumns() {
		return featureCount;
	}

	@Override
	public void reset() {
        exampleStartOffsets.clear();
		initializeOffsets();
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor arg0) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not implemented");
	}
	
	@Override
	public int totalExamples() {
		return (sampleCount-1) / batchSize - 2;
	}
	
	@Override
	public int numExamples() {
		return totalExamples();
	}

}
