package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
 * @author rockt
 */
/**
 * Example training of a model
 *
 * Problems 2/3/4: perform a grid search over the parameters below
 */
object Main extends App {
  /**
   * Example training of a model
   *
   * Problems 2/3/4: perform a grid search over the parameters below
   */

  val trainSetName = "train"
  val validationSetName = "dev"


  //11) 75.37 (7,1.0E-5,0.1)1 iter => deeper
//  10,  0.1,  0.0031622776601683794
//  10, 0.01, 0.01, 83.65680110666567, 76.36484687083887
//  example of making model and storing it
//  val testModel = new SumOfWordVectorsModel(10, 0.01)
//  val testParamsLogString = (10, 0.01, 0.01).productIterator.toList.mkString(" ")
//  StochasticGradientDescentLearner(testModel, trainSetName, 10, 0.01, isEarlyStop=true, testParamsLogString, "test_sumofword_run_param_history.txt")
//  val result = testModel.predict("doesn't not enjoy re-piercing ears".split(" "))
//  println("result of loaded model is "+result)
//  SaveModel.saveModelToFile(testModel, "test_sumofword_model.txt")
//  example of loading model
//  SaveModel.loadModelFromFile("Wed Jan 27 13:53:05 GMT 2016test_sumofword_model.txt")
//  val loadedModel = new SumOfWordVectorsModel(10, 0.01)
//  val loadedResult = loadedModel.predict("doesn't not enjoy re-piercing ears".split(" "))
//  println("result of loaded model is "+loadedResult)


// q. 4.3.4)
//  val wordDimSet = 10 to 11 by 1
//  val vectorRegStrengthSet = (-6.0 to 0.0 by 1.0).map(a => Math.pow(10,a))
//  val learningRateSet = (-6.0 to 0.0 by 0.5).map(a => Math.pow(10,a))
//  runGridSearch(wordDimSet, vectorRegStrengthSet, learningRateSet, 100)
  // prints out best set of hyperparameters based on 6th log in the file
//  SaveModel.printBestParamFromFile("sumofword_grid_search_param_history.txt", 6)
//  ToDo we can visualize parameter space i.e. 3d graph of params and validation set


// q. 4.3.5)
//  val wordDim = 8
//  val vectorRegularizationStrength = 0.001
//  val learningRate = 0.01
//  iterateThroughEpochs(wordDim, vectorRegularizationStrength, learningRate)


// q. 4.3.6) prints vectorparams to file to visualize them later
//  val sumOfWordModel = new SumOfWordVectorsModel(10, 0.01)
//  val sumOfWordParamsLogString = (10, 0.01, 0.01).productIterator.toList.mkString(" ")
//  StochasticGradientDescentLearner(sumOfWordModel, trainSetName, 100, 0.01, isEarlyStop=true, sumOfWordParamsLogString, "sumofword_run_param_history.txt")
//  SaveModel.writeWordVectorsFromModelToFile(sumOfWordModel, 500)


// q. 4.4.2) run norm SGDL with RNN model for debug
  val RNNmodel: Model = new RecurrentNeuralNetworkModel(7, 3, 0.00001, 0.0001)
//  val rnnParamsLogString = (10, 10, 0.01, 0.01, 0.01).productIterator.toList.mkString(" ")
  StochasticGradientDescentLearner(RNNmodel, trainSetName, 100, 0.001, isEarlyStop=true, "", "rnn_run_param_history.txt")


  val wordDimRange = 7 to 11 by 2
  val hiddenDimRange = 3 to 7 by 2
  val vectorRegStrengthRange = (-5.0 to -1.0 by 1.0).map(a => Math.pow(10,a))
  val matrixRegStrengthRange = (-4.0 to -1.0 by 1.0).map(a => Math.pow(10,a)) :+ 0.0
  val learningRateRange = IndexedSeq(0.001, 0.01, 0.03, 0.1) //  (-3.0 to -1.0 by 1.0).map(a => Math.pow(10,a))
//  runGridSearchRNN(wordDimRange, hiddenDimRange, vectorRegStrengthRange, matrixRegStrengthRange, learningRateRange, 100)
//  SaveModel.printBestParamFromFile("rnn_run_param_history.txt")


//  val mulOfWordModel = new MulOfWordsModel(10, 0.005)
//  StochasticGradientDescentLearner(mulOfWordModel, trainSetName, 100, 0.01, isEarlyStop=true, "", "bullshit.txt")


// q. 4.5)
//  val lstmWordDim = 8
//  val lstmHiddenDim = 3
//  val lstmVectorRegulStrength = 0.001
//  val lstmMatrixRegulStrength = 0.001
//
//  val LSTMModel = new LSTMModel(lstmWordDim, lstmHiddenDim, lstmVectorRegulStrength, lstmMatrixRegulStrength)
//  val lstmParamsLogString = (10, 0.01, 0.01).productIterator.toList.mkString(" ")
//  StochasticGradientDescentLearner(LSTMModel, trainSetName, 100, learningRate, isEarlyStop=true, lstmParamsLogString, "lstm_run_param_history.txt")


  def runGridSearch(wordDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {
    for (wordDim <- wordDimSet; vectorRegStrength <- vectorRegStrengthSet; learningRate <- learningRateSet) {
        LookupTable.trainableWordVectors.clear()
        val paramsLogString = "wordDim="+wordDim+" vectorReg="+vectorRegStrength+" learningRate="+learningRate
        val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegStrength)

        StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, isEarlyStop=true, paramsLogString, "sumofword_grid_search_param_history.txt")
    }
  }


  def iterateThroughEpochs(wordDim:Int, vectorRegularizationStrength:Double, learningRate:Double): Unit = {
    val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
    val paramsLogString = (wordDim, vectorRegularizationStrength, learningRate).productIterator.toList.mkString(" ")

    StochasticGradientDescentLearner(gridSearchModel, trainSetName, 100, learningRate, isEarlyStop=false, paramsLogString, "epoch_iteration_param_history.txt")
  }


  def runGridSearchRNN(wordDimRange:Range, hiddenDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], matrixRegStrengthSet:IndexedSeq[Double],
                       learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {
    for (wordDim <- wordDimRange; hiddenDim <- hiddenDimSet; vectorRegStrength <- vectorRegStrengthSet; matrRegS <- matrixRegStrengthSet; learningRate <- learningRateSet) {
        LookupTable.trainableWordVectors.clear()
        val paramsLogString = (wordDim, hiddenDim, vectorRegStrength, matrRegS, learningRate).productIterator.toList.mkString(" ")
        val gridSearchModel = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegStrength, matrRegS)

        StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, isEarlyStop=true, paramsLogString, "rnn_grid_search_param_history.txt")
    }
  }



  /**
   * Comment this in if you want to look at trained parameters
   */
  /*
  for ((paramName, paramBlock) <- model.vectorParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  */
}