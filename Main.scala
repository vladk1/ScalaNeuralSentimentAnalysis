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

  val trainSetName = "train" // "debug"
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

  val mulOfWordModel = new MulOfWordsModel(11, 0.001)
//  val mulOfWordParamsLogString = (11, 0.001, 0.03).productIterator.toList.mkString(" ")
  StochasticGradientDescentLearner(mulOfWordModel, trainSetName, 100, 0.03, isEarlyStop=true, "", "mulOfWordModelLog.txt")

  val wordDimSet = 6 to 11 by 1
  val vectorRegStrengthSet = (-5.0 to 0.0 by 1.0).map(a => Math.pow(10,a))
  val learningRateSet = (-5.0 to 0.0 by 0.5).map(a => Math.pow(10,a))
//  val wordDim = 10
//  val regStrength = 0.01
//  val learningRate = 0.01
//  val mulOfWordModel = new MulOfWordsModel(wordDim, regStrength)
//  val mulOfWordParamsLogString = s"wordDim=$wordDim regStrength=$regStrength learningRate=$learningRate"
//  StochasticGradientDescentLearner(mulOfWordModel, trainSetName, 100, 0.01, isEarlyStop=true, mulOfWordParamsLogString, "mulOfWordModelLog.txt")

//  val wordDimSet = 6 to 11 by 1
//  val vectorRegStrengthSet = (-5.0 to 0.0 by 1.0).map(a => Math.pow(10,a))
//  val learningRateSet = (-5.0 to 0.0 by 0.5).map(a => Math.pow(10,a))

//  runGridSearchOnMultOfWord(wordDimSet, vectorRegStrengthSet, learningRateSet, 100)

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
//  val RNNmodel: Model = new RecurrentNeuralNetworkModel(7, 3, 0.0001, 0.0001)
//  val rnnParamsLogString = (10, 10, 0.01, 0.01, 0.01).productIterator.toList.mkString(" ")
//  StochasticGradientDescentLearner(RNNmodel, trainSetName, 100, 0.001, isEarlyStop=true, "", "rnn_run_param_history.txt")
  //  Epoch   21	Dev_Loss 765.38	Dev_Acc 77.50
  //  9 7 0.001 0.001 0.001
//  val RNNmodel: Model = new RecurrentNeuralNetworkModel(9, 7, 0.001, 0.001)
//  val rnnParamsLogString = (9, 7, 0.001, 0.001).productIterator.toList.mkString(" ")
//  StochasticGradientDescentLearner(RNNmodel, trainSetName, 100, 0.001, isEarlyStop=true, "", "rnn_run_param_history.txt")

//    LookupTable.trainableWordVectors.clear()
//    println("run LSTM train:dev")
//    val LSTMModel = new LSTMModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
//    StochasticGradientDescentLearner(LSTMModel, trainSetName, 100, learningRate, isEarlyStop = true, "", "tt.txt")

  // Tim settings:
  // init learning rate [1E-4, 3E-4, 1E-3]
  // dropout [0.0, 0.1, 0.2]
  // reg [0.0, 1E-4, 3E-4, 1E-3]

//  val wordDimRange = 7 to 11 by 2
//  val hiddenDimRange = 7 to 11 by 2
//  val regStrengthRange = IndexedSeq(0.0, Math.pow(10,-4), 3*Math.pow(10,-4), Math.pow(10,-3))
//  val matrixRegStrengthRange = IndexedSeq(Math.pow(10,-4), 0.001)

//  best: Epoch   21	Dev_Loss 765.38	Dev_Acc 77.50 9 7 0.001 0.001 0.001
//  val learningRateRange = IndexedSeq(0.001)
//  runGridSearchRNN(wordDimRange, hiddenDimRange, regStrengthRange, regStrengthRange, learningRateRange, 100)
//  SaveModel.printBestParamFromFile("rnn_run_param_history.txt")

//  val mulOfWordModel = new MulOfWordsModel(10, 0.005)
//  StochasticGradientDescentLearner(mulOfWordModel, trainSetName, 100, 0.01, isEarlyStop=true, "", "bullshit.txt")



// q. 4.5)
  val lstmWordDim = 10
  val lstmHiddenDim = 10
  val lstmVectorRegulStrength = 0.001
  val lstmMatrixRegulStrength = 0.001
  val lstmLearningRate = 0.05
  val LSTMModel = new LSTMModel(lstmWordDim, lstmHiddenDim, lstmVectorRegulStrength, lstmMatrixRegulStrength)

  val lstmParamsLogString = (lstmWordDim, lstmHiddenDim, lstmVectorRegulStrength, lstmLearningRate).productIterator.toList.mkString(" ")
  StochasticGradientDescentLearner(LSTMModel, trainSetName, 100, lstmLearningRate, isEarlyStop=true, lstmParamsLogString, "lstm_run_param_history.txt")

//  runGridSearch(wordDimRange, vectorRegStrengthRange, learningRateRange, 100)

  def runGridSearch(wordDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {
    for (wordDim <- wordDimSet; vectorRegStrength <- vectorRegStrengthSet; learningRate <- learningRateSet) {
        LookupTable.trainableWordVectors.clear()
        val paramsLogString = "wordDim="+wordDim+" vectorReg="+vectorRegStrength+" learningRate="+learningRate
        val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegStrength)
        StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, isEarlyStop=true, paramsLogString, "sumofword_grid_search_param_history.txt")
    } // 100 27 381
  }

//  multiplication of word vectors model
  def runGridSearchOnMultOfWord(wordDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {
    for (wordDim <- wordDimSet; vectorRegStrength <- vectorRegStrengthSet; learningRate <- learningRateSet) {
      LookupTable.trainableWordVectors.clear()
      val paramsLogString = "wordDim="+wordDim+" vectorReg="+vectorRegStrength+" learningRate="+learningRate
      val gridSearchModel = new MulOfWordsModel(wordDim, vectorRegStrength)
      StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, isEarlyStop=true, paramsLogString, "multOfWord_grid_search_param_history.txt")
    }
  }


  def iterateThroughEpochs(wordDim:Int, vectorRegularizationStrength:Double, learningRate:Double): Unit = {
    val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
    val paramsLogString = (wordDim, vectorRegularizationStrength, learningRate).productIterator.toList.mkString(" ")

    StochasticGradientDescentLearner(gridSearchModel, trainSetName, 100, learningRate, isEarlyStop=false, paramsLogString, "epoch_iteration_param_history.txt")
  }


  def runGridSearchRNN(wordDimRange:Range, hiddenDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], matrixRegStrengthSet:IndexedSeq[Double],
                       learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {
    for (wordDim <- wordDimRange; hiddenDim <- hiddenDimSet; regStrength <- vectorRegStrengthSet;  learningRate <- learningRateSet) {
        LookupTable.trainableWordVectors.clear()
        val paramsLogString = (wordDim, hiddenDim, regStrength, regStrength, learningRate).productIterator.toList.mkString(" ")
        val gridSearchModel = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, regStrength, regStrength)

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