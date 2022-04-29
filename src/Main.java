import java.util.List;
import java.util.Random;

public class Main {
	public final static int NUM_HIDDENLAYERS = 2;
	public final static int NUM_NODES = 32;
	public final static int BATCH_SIZE = 100;
	public final static int REPETITIONS = 100;
	
	public final static double STDEV = 1e-6;
	public final static int WEIGHT_RANGE = 1;
	public final static int BIAS_RANGE = 1;
	
	public final static double LEARNING_RATE = 0.1;
	public final static Random RAND = new Random();
	
	public static Node[][] node;
	public static int target;
	
	public Main(int nHiddenLayers, int nNodes, int batchSize, int repetitions) {
		//SETUP
		int[] labels = MnistReader.getLabels("C:\\Users\\JakeTheSpectre\\eclipse-workspace\\Neural\\data\\training.labels");
		List<int[][]> images = MnistReader.getImages("C:\\Users\\JakeTheSpectre\\eclipse-workspace\\Neural\\data\\training.images");
		System.out.println("Loaded Images");
		node = new Node[4][];
		node[0] = FileIO.convertToNode(images.get(0));
		node[1] = new Node[nNodes];
		node[2] = new Node[nNodes];
		node[3] = new Node[10];
		//Set random values for data nodes
		for(int i = 1; i < node.length; i++) {
			for(int j = 0; j < node[i].length; j++) {
				int nodesInPrev = i==0 ? 0 : node[i-1].length;
				node[i][j] = new Node(i,j,nodesInPrev);
			}
		}
		int numBatches = 60000/batchSize;
		int target = 0;
		for(int r = 0; r < repetitions; r++) {
			double correct = 0;
			for(int m = 0; m < numBatches; m++) {
				for(int n = 0; n < batchSize; n++) {
					//Load in image
					node[0] = FileIO.convertToNode(images.get(n+m*batchSize));
					target = labels[n+m*batchSize];
//					System.out.println(target);
//					for(int i = 1; i < node[0].length; i++) {
////						String s = (int)Math.floor(node[0][i].activation*10)==0 ? " " : ((int)Math.floor(node[0][i].activation*10)-1)+"";
//						String s = Double.toString(node[0][i].activation);
//						System.out.print(s+" ");
//						if(i%28==0){ System.out.println();}
//					}
//					System.out.println();
					
					
					//Process nodes activations (this could probably be done in the last set of for loops to save on looping)
					for(int i = 1; i < node.length; i++) {
						for(int j = 0; j < node[i].length; j++) {
							node[i][j].activate();
						}
					}
					
					//Check for correct answer
					int guess = -1;
					double gActivation = -1;
					for(int j = 0; j < node[3].length; j++) {
//						if(j==3) {System.out.println("Guess: "+j+" Activation: "+node[3][j].bias);}
						if(node[3][j].activation >= gActivation) {
//							System.out.println("Switched");
							guess = j;
							gActivation = node[3][j].activation;
						}
					}
					if(guess == target) {
//						System.out.println(guess);
						correct++;
					}
					
					//Set all node's deltas
					for(int i = node.length-1; i > 0; i--) {
						for(int j = 0; j < node[i].length; j++) {
							if(i == node.length-1) {
								node[i][j].setDelta(j == target ? 1 : 0);
							} else {
								node[i][j].setDelta();
							}
						}
					}
					//Apply backpropagation for each node
					for(int i = 1; i < node.length; i++) {
						for(int j = 0; j < node[i].length; j++) {
							node[i][j].save();
						}
					}
					//Clear activation, delta, and cWeights
					//Or don't? They'll just get overwritten
				}
//				System.out.println("NEW BATCH");
				//Change all biases and weights to new values (step 11)
				//Clear tDelta and tWeights while i'm at it
				for(int i = 1; i < node.length; i++) {
					for(int j = 0; j < node[i].length; j++) {
						node[i][j].backpropagate();
					}
				}
				//Continue to the next batch
			}
			System.out.println("Repetition #"+r);
			System.out.println("# Correct:"+correct);
			System.out.println("% Correct:"+Math.round(correct*100*100/60000)/100.0);
		}
		//Now test for accuracy
		System.out.println("Testing...");
		labels = MnistReader.getLabels("C:\\Users\\JakeTheSpectre\\eclipse-workspace\\Neural\\data\\testing.labels");
		images = MnistReader.getImages("C:\\Users\\JakeTheSpectre\\eclipse-workspace\\Neural\\data\\testing.images");
		int correct = 0;
		int numOfTestingFiles = 10000;
		for(int n = 0; n < numOfTestingFiles; n++) {
			node[0] = FileIO.convertToNode(images.get(n));
			target = labels[n];
			//Process nodes activations
			for(int i = 1; i < node.length; i++) {
				for(int j = 0; j < node[i].length; j++) {
					node[i][j].activate();
				}
			}
			//Check Guess
			int guess = -1;
			double gActivation = -1;
			for(int j = 0; j < node[3].length; j++) {
				if(node[3][j].activation >= gActivation) {
					guess = j;
					gActivation = node[3][j].activation;
				}
			}
			if(guess == target) {
				correct++;
			}
		}
		System.out.println("NumCorrect: "+correct);
		System.out.println("%Correct: "+correct/100.0);
	}
	
	public static void main(String[] args) {
		new Main(NUM_HIDDENLAYERS,NUM_NODES,BATCH_SIZE,REPETITIONS);
	}
}
