import java.util.Random;

public class Node {
	int layerNumber;
	int indexNumber;
	double[] weights;
	double bias;
	double activation;
	double z;
	
	double delta;
	double tBias;
	double[] tWeights;
	public Node(int lNum, int iNum, double a) {
		this.layerNumber = lNum;
		this.indexNumber = iNum;
		this.activation = a;
	}
	public Node(int lNum, int iNum, int nNodesInPrevLayer) {
		this.layerNumber = lNum;
		this.indexNumber = iNum;
		weights = new double[nNodesInPrevLayer];
		tWeights = new double[nNodesInPrevLayer];
		for(int k = 0; k < weights.length; k++) {
			//weights[k] = (Math.random()-.5)*Main.WEIGHT_RANGE;
			weights[k] = Main.RAND.nextGaussian() * Main.STDEV;
			tWeights[k] = 0;
		}
		bias = Main.RAND.nextGaussian() * Main.STDEV;
		z = 0;
		delta = 0;
		tBias = 0;
	}
	
	public void backpropagate() {
		this.bias = newBias(this.bias,Main.LEARNING_RATE,this.tBias/100.0);
		this.tBias = 0;
		for(int i = 0; i < weights.length; i++) {
			this.weights[i] = newWeight(this.weights[i],Main.LEARNING_RATE,this.tWeights[i]/100.0);
			this.tWeights[i] = 0;
		}
	}
	
	public void save() {
		this.tBias += this.delta;
		for(int i = 0; i < weights.length; i++) {
			this.tWeights[i] += dCdW_Del(Main.node[this.layerNumber-1][i].activation);
		}
	}
	
	public static double dCdA(double target, double activation) {
		return 2*(target-activation);
	}
	public static double dAdZ(double activation) {
		return activation*(1-activation);
	}
	public static double dZdW(double pActivation) {
		return pActivation;
	}
	public static double delta(double target, double activation) {
		return dCdA(target,activation)*dAdZ(activation);
	}
	public static double dCdW(double target,double activation, double pActivation) {
		return delta(target,activation)*dZdW(pActivation);
	}
	public static double newWeight(double curWeight, double learningRate, double dCdW) {
		return curWeight-learningRate*dCdW;
	}
	public static double newBias(double curBias, double learningRate, double delta) {
		return curBias-learningRate*delta;
	}
	public static double getDCDNA(int nextLayer, int curIndex) {
		//This only works if delta is already set for the nextLayer's node
		double d = 0;
		for(int i = 0; i < Main.node[nextLayer].length; i++) {
			d += Main.node[nextLayer][i].getDC0DNA_Del(curIndex);
		}
		return d;
	}
	public static double delta(int curLayer, int curIndex, double activation) {
		return getDCDNA(curLayer+1,curIndex)*dAdZ(activation);
	}

	//These ones can be used if delta is already set
	public double dCdW_Del(double pActivation) {
		return this.getDelta()*dZdW(pActivation);
	}
	public double newWeight_Del(double curWeight, double learningRate, double pActivation) {
		return curWeight-learningRate*dCdW_Del(pActivation);
	}
	public double newBias_Del(double curBias, double learningRate) {
		return curBias-learningRate*this.getDelta();
	}
	public double getDC0DNA_Del(int pNodeIndex) {
		return this.getDelta()*this.getWeight(pNodeIndex);
	}
	
	
	public double getDelta() {
		return delta;
	}
	public double getWeight(int index) {
		return weights[index];
	}
	public void setDelta(double target) {
		delta = delta(target,this.activation);
	}
	public void setDelta() {
		delta = delta(this.layerNumber,this.indexNumber,this.activation);
	}
	
	
	public void activate() {
		this.z = zValue();
		this.activation = sigmoid(this.z);
	}
	public double zValue() {
		double z = bias;
		for(int j = 0; j < weights.length; j++) {
			z += weights[j]*Main.node[layerNumber-1][j].activation;
		}
		return z;
	}
	public double sigmoid(double z) {
		return 1/(1+Math.pow(Math.E, -z));
	}
}

/*
 * 
 * 16(784)    16(16)    10(16)
 * 
 * 1. Calc partialCpartialA for output nodes
 * 
 * Nodes in (Layer-1) will have an array of weightmultipliers it gets from Nodes in (Layer) for the first part of backprop.
 * Each node in (Layer-1) should have:
 * 		Array of Weights relating to (Layer-2)
 * 		Array of partialCpartialA from (Layer)
 * 		SquishPrime(zValue)
 * And then when I call calcCost() it should:
 * 		Get partialCpartialA array and add them
 * 		Set weightMultiplier to AvgPartialCPartialA*Activation
 * 		Set cBias as weightMultiplier
 * 		Make an Array of cWeights (to change weight array) by:
 * 			Doing Weights * weightMultiplier
 * 
 * 		Then it needs to further backpropagate.
 * 		Calculate partialCpartialA
 * 
 * 		
 * C = (a(L)-y)^2
 * z = w(L)a(L-1)+b(L)
 * a = squish(z)
 * 
 * 
 * partialC/partialB = 
 * 		partialZ/partialB = 1
 * 		partialA/partialZ = squishPrime(z)
 * 		partialC/partialA = either:
 * 								error between current activation and what it should be
 * 								Sum(Array<partialCpartialA>) from Layer+1
 * 
 * Array<partialC/partialW> = 
 * 		partialZ/partialW = Array<Activations(Layer-1)>
 * 		partialA/partialZ = squishPrime(z)
 * 		partialC/partialA = either:
 * 								error between current activation and what it should be
 * 								Sum(Array<partialCpartialA>) from Layer+1
 * 
 * partialC/partialA(Layer-1) = 
 * 		partialZ/partialA(Layer-1) = Sum(Array<weight(Layer)>)
 * 		partialA/partialZ = squishPrime(z)
 * 		partialC/partialA = either:
 * 								error between current activation and what it should be
 * 								delta(Layer)*Sum(Array<weight(Layer)>)
 * 
 * To do good backprop:
 * 		1. for each node in last layer set partialCpartialA equal to error
 * 		2. for each node set delta = partialCpartialA*squishPrime(z)
 * 		3. for each node set Array<cWeights> = delta*Array<Activations(Layer-1)>
 * 
 * 		4. for each node in (Layer-1) get Array<partialCpartialA> = Array<delta(Layer)> ** Array<weight(Layer)>
 * 				then set partialCpartialA = Avg(Array<partialCpartialA>)
 * 		5. for each node in (Layer-1) set delta
 * 		6. for each node in (Layer-1) set Array<cWeights>
 * 
 * 		7. for each node in (Layer-2) set partialCpartialA
 * 		8. for each node in (Layer-2) set delta
 * 		9. for each node in (Layer-2) set Array<cWeights>
 * 
 * 		10. Add all deltas and cWeights to totaldelta and totalcWeight
 * 		11. Do another training example and instead of setting delta, add to delta (Repeat 1-9 100x, once for each image in batch)
 * 		12. Repeat steps 1-11 600x, once for each batch.
 * 		
 */











