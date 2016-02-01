package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.FileWriter

/**
 * @author rockt
 */
object Main extends App {

  val trainSetName = "train"
  val validationSetName = "dev"
  val testSetName = "test"

  /**********************************************************************************
                       USE THESE METHODS AS A WAY TO RUN
                               EACH QUESTION
                 1 question2()  - SumOfWordVectorsModel
                 2 question3() - RNN
                 3 question4() - SkyIsTheLimit (a) MulOfWords (b) LSTM (WARNING: Slow!)
    *********************************************************************************/
//  question2()
//  question3()
//  question4()

  def question2()  {

    /**********************************************************************************
                        q 4.3.2 Load best model from file
      *********************************************************************************/
//      If best model file is added to data directory uncomment this section to load pre-trained sum of words
//      File was provided as part of submission
//    val loadedBestSumOfWordModel = loadBestSumOfWordVectorModel("best_sumofword_model_vectorsMonFeb01204132GMT2016.txt", 10)
//    val evaluatorOnValidSetFromFile = Evaluator(loadedBestSumOfWordModel, Main.validationSetName)
//    val devAccFromFile = evaluatorOnValidSetFromFile._1
//    val devLossFromFile = evaluatorOnValidSetFromFile._2
//    println(s"Dev_Loss=$devLossFromFile Dev_Acc=$devAccFromFile ")

    val wordDim = 10
    val regStrength = 0.1
    val learningRate = 0.01
    val sumOfWordsModel = new SumOfWordVectorsModel(wordDim, regStrength)
    val sumOfWordParamsLogString = s"wordDim=$wordDim regStrength=$regStrength learningRate=$learningRate"
    StochasticGradientDescentLearner(sumOfWordsModel, trainSetName, 100, learningRate, isEarlyStop=true, sumOfWordParamsLogString, "sumOfWordsSGD.txt")

    // print test predictions for SumOfWords model
    get_predictions(sumOfWordsModel, testSetName, "predictions.txt")

    /**********************************************************************************
                          q 4.3.4 Grid Search for Sum of Words
     *********************************************************************************/
    val wordDimSet = 8 to 14 by 2
    val vectorRegStrengthSet = (-5.0 to -1.0 by 1.0).map(a => Math.pow(10,a))
    val learningRateSet = (-4.0 to 0.0 by 0.5).map(a => Math.pow(10,a))
    runGridSearch(wordDimSet, vectorRegStrengthSet, learningRateSet, 100, "sumofword_grid_search_param_history.txt")
    SaveModel.printBestParamsFromLogFile("sumofword_grid_search_param_history.txt", "Dev_Acc", minimize = false)
    // We obtained the best model with parameters:
    // Epoch 18 wordDim=10 vectorReg=0.1 learningRate=0.01 iter=18 trainLoss=61503.20382496397 Dbest_sumofword_model_run_param_historyev_Loss=931.3517936096621 Dev_Acc=77.96271637816245
    val bestSumOfWordModel = trainBestSumOfWordVectorsModel(10, 0.1, 0.01, "best_sumofword_model_run_param_history.txt", isEarlyStop = true, 100)
    SaveModel.printBestParamsFromLogFile("best_sumofword_model_run_param_history.txt", "Dev_Acc", minimize = false)

    /**********************************************************************************
                              q 4.3.5 Loss Analysis
      *********************************************************************************/
    val iterateThroughSumOfWordModel = trainBestSumOfWordVectorsModel(10, 0.1, 0.01, "epoch_iteration_param_history.txt", isEarlyStop = false, 100)

    /**********************************************************************************
                  q 4.3.6 prints vectorparams to file to visualize them later
      *********************************************************************************/
//    SaveModel.writeWordVectorsFromModelToFile(bestSumOfWordModel, 100, "actual_word_param100.txt", "word_param100.txt", "vector_params100.txt")
  }

  def question3():Unit = {
    /**********************************************************************************
                        q 4.4.2 Run RNN with best parameters
      *********************************************************************************/
    //  We obtained the best model with parameters:
    //  wordDim=11 hiddenDim=11 regStrength=1.0E-4 learningRate=0.001 iter=20 trainLoss=51394.85565085464 Dev_Loss=775.6154914921979 Dev_Acc=77.29693741677764
    val bestRNN = trainBestRNNModel(11, 11, 1.0E-4, 1.0E-4, 0.001, "best_rnn_model_run_param_history.txt", isEarlyStop = false)
    SaveModel.printBestParamsFromLogFile("best_rnn_model_run_param_history.txt", "Dev_Acc", minimize = false)
    SaveModel.saveModelToFile(bestRNN, "rnn_model_with_stepdecay1_vectors.txt")

    /**********************************************************************************
                        q 4.4.3 Run RNN Grid Search
      *********************************************************************************/
    val wordDimRange = IndexedSeq(7,11)
    val hiddenDimRange =  IndexedSeq(9, 11)
    val regStrengthRange = IndexedSeq(0.0, Math.pow(10,-4), Math.pow(10,-3.5), Math.pow(10,-3))
    val learningRateRange = IndexedSeq(0.001, 0.01)
//    runGridSearchRNN(wordDimRange, hiddenDimRange, regStrengthRange, regStrengthRange, learningRateRange, 100, "rnn_grid_search_param_history.txt")
//    SaveModel.printBestParamsFromLogFile("rnn_grid_search_param_history.txt", "Dev_Acc", minimize = false)
  }

  def question4():Unit = {

    /**********************************************************************************
                       q 4.5 Run MulSum of Words Model with best parameters
      *********************************************************************************/
    val wordDim = 11
    val regStrength = 0.001
    val learningRate = 0.03
    val mulOfWordModel = new MulOfWordsModel(11, regStrength)
    val mulOfWordParamsLogString = s"wordDim=$wordDim regStrength=$regStrength learningRate=$learningRate"
    StochasticGradientDescentLearner(mulOfWordModel, trainSetName, 100, learningRate, isEarlyStop=true, mulOfWordParamsLogString, "mulOfWordModelLog.txt")

    // print test predictions for mulOfWords model
    get_predictions(mulOfWordModel, testSetName, "predictions_own.txt")

    /**********************************************************************************
                        Run Grid Search for MulSum of Words Model
      *********************************************************************************/
    val wordDimSet = 8 to 11 by 1
    val vectorRegStrengthSet = (-4.0 to -1.0 by 1.0).map(a => Math.pow(10,a))
    val learningRateSet = (-4.0 to -1.0 by 0.5).map(a => Math.pow(10,a))
//    runGridSearchOnMultOfWord(wordDimSet, vectorRegStrengthSet, learningRateSet, 100)

    /**********************************************************************************
                                      Run LSTM Model
      *********************************************************************************/
    val lstmWordDim = 10
    val lstmHiddenDim = 10
    val lstmVectorRegulStrength = 0.001
    val lstmMatrixRegulStrength = 0.001
    val lstmLearningRate = 0.05
    val LSTMModel = new LSTMModel(lstmWordDim, lstmHiddenDim, lstmVectorRegulStrength, lstmMatrixRegulStrength)

    val lstmParamsLogString =  s"lstmWordDim=$lstmWordDim lstmHiddenDim=$lstmHiddenDim lstmVectorRegulStrength=$lstmVectorRegulStrength lstmLearningRate=$lstmLearningRate"
//    StochasticGradientDescentLearner(LSTMModel, trainSetName, 100, lstmLearningRate, isEarlyStop=true, lstmParamsLogString, "lstm_run_param_history.txt")
  }


  def runGridSearch(wordDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], learningRateSet:IndexedSeq[Double], epochs:Int, outputFile:String): Unit = {
    for (wordDim <- wordDimSet; vectorRegStrength <- vectorRegStrengthSet; learningRate <- learningRateSet) {
      LookupTable.trainableWordVectors.clear()
      val paramsLogString = s"wordDim=$wordDim vectorReg=$vectorRegStrength learningRate=$learningRate"
      val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegStrength)
      StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, isEarlyStop=true, paramsLogString, outputFile)
    }
  }

  def trainBestSumOfWordVectorsModel(embedSize:Int, regStrength:Double, learningRate:Double, outputFile:String, isEarlyStop:Boolean, maxEpoch:Int):Model = {
    val testModel = new SumOfWordVectorsModel(embedSize, regStrength)
    val testParamsLogString = s"embeddingSize=$embedSize regStrength=$regStrength learningRate=$learningRate"
    StochasticGradientDescentLearner(testModel, trainSetName, maxEpoch, learningRate, isEarlyStop, testParamsLogString, outputFile)
    testModel
  }

  def loadBestSumOfWordVectorModel(modelFileName:String, embeddingSize:Int): Model = {
    val loadedModel = new SumOfWordVectorsModel(embeddingSize, 0) // params doesn't matter as weights are updated already, except expectedEmbedSize
    SaveModel.loadModelFromFile(modelFileName)
    loadedModel
  }

  def trainBestRNNModel(embedSize:Int, hiddenSize: Int, vecRegStrength:Double, matRegStrength:Double, learningRate:Double, outputFile:String, isEarlyStop:Boolean):Model = {
    val bestRNNmodel: Model = new RecurrentNeuralNetworkModel(embedSize, hiddenSize, vecRegStrength, matRegStrength)
    val rnnParamsLogString = s"embeddingSize=$embedSize hiddenSize=$hiddenSize vecRegStrength=$vecRegStrength matRegStrength=$matRegStrength learningRate=$learningRate"
    StochasticGradientDescentLearner(bestRNNmodel, trainSetName, 100, learningRate, isEarlyStop, rnnParamsLogString, outputFile)
    bestRNNmodel
  }

  def runGridSearchRNN(wordDimRange:IndexedSeq[Int], hiddenDimSet:IndexedSeq[Int], vectorRegStrengthSet:IndexedSeq[Double], matrixRegStrengthSet:IndexedSeq[Double],
                       learningRateSet:IndexedSeq[Double], epochs:Int, outputFile:String): Unit = {
    for (wordDim <- wordDimRange; hiddenDim <- hiddenDimSet; regStrength <- vectorRegStrengthSet; learningRate <- learningRateSet) {
      LookupTable.trainableWordVectors.clear()
      val paramsLogString = s"wordDim=$wordDim hiddenDim=$hiddenDim regStrength=$regStrength learningRate=$learningRate"
      val gridSearchModel = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, regStrength, regStrength)
      StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, isEarlyStop=true, paramsLogString, outputFile)
    }
  }

  def runGridSearchOnMultOfWord(wordDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {
    for (wordDim <- wordDimSet; vectorRegStrength <- vectorRegStrengthSet; learningRate <- learningRateSet) {
      LookupTable.trainableWordVectors.clear()
      val paramsLogString = "wordDim="+wordDim+" vectorReg="+vectorRegStrength+" learningRate="+learningRate
      val gridSearchModel = new MulOfWordsModel(wordDim, vectorRegStrength)
      StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, isEarlyStop=true, paramsLogString, "multOfWord_grid_search_param_history.txt")
    }
  }

  // prints to file the predicted sentiment (1 or 0) for each sentence in corpus
  def get_predictions(model: Model, corpus: String, fileName: String): Unit ={
    val predictionWriter = new FileWriter("./data/assignment3/" + fileName)
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    for(iter <- 0 until iterations){
      val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
      predictionWriter.write(BoolToSentiment(model.predict(sentence)) + "\n")
    }
    predictionWriter.close()
  }

  def BoolToSentiment(bool: Boolean): Int ={
    bool match{
      case false => 0
      case true => 1
    }

  }
}