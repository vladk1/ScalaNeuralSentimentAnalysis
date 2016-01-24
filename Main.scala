package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{File, PrintWriter}

import scala.collection.immutable.NumericRange
import scala.collection.mutable
import scala.util.control.Breaks

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

//    Using for RNN
  val learningRate = 0.01 // tried 0.01, 0.1, 1.0, 10.0, 1.0,
  val vectorRegularizationStrength = 0.01 // tried 0.01, 0.1, 0.01, _, _, 5.0
  val matrixRegularizationStrength = 0.0 // tried 0.01, 0.1, 0.01, _, _, 5.0
  val wordDim = 10 // tried 10, 5
  val hiddenDim = 10 // tried 10, 5

  val trainSetName = "train"
  val validationSetName = "dev"

//  val trainSetName = "debug"
//  val validationSetName = "dev"

  val wordDimSet = 5 to 13 by 1
  val vectorRegStrengthSet = 0.001 to 0.015 by 0.005 // in case Nan - higher regularizer
  // in case Nan - lower learning rate (hence we can stop iterating after reaching Nan in this loop)
  val learningRateSet = 0.01 to 0.99 by 0.01

  //  1) 75.43  (6, 0.45, 0.06) with 1 iteration:
  //  2) 77.16  (7, 0.1, 0.05) -//-
  //  3) 76.298 (5, 0.05, 0.045) -//-
  //  4) 75.433 (6, 0.01, 0.015) with better tokenizer (gives us 225 998 words vs old 275 826)
  //  5) 75.099 (7, 0.01, 0.01) 10 iter
  //  6) 75.433 (6, 0.001, 0.001) 10 iter
  //  runs grid search and prints best params:
  /**
   * Grid Search
   */
//    runGridSearch(wordDimSet, vectorRegStrengthSet, learningRateSet, 1)
  /**
    * SGDL
    */
  // run norm SGDL with RNN model for debug
  val SGDLmodel = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
  StochasticGradientDescentLearner(SGDLmodel, trainSetName, 100, learningRate, epochHook)


//  prints vectorparams to file
//  writeVectorsFromBestModelToFile(7, 0.1, 0.05, 10)

  def epochHook(iter: Int, accLoss: Double, model: Model): Unit = {
    println("Epoch %4d\tLoss %8.4f\tTrain_Acc %4.4f\tDev_Acc %4.4f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100 * Evaluator(model, validationSetName)))
  }

  def hasExplodingGradient(model: Model): Boolean = {
    val (sentence, target) = SentimentAnalysisCorpus.getExample(trainSetName)
    val modelLoss = model.loss(sentence, target)
    modelLoss.forward().isNaN
  }

  def writeVectorsFromBestModelToFile(bestWordDim:Int, bestVectorRegularizationStrength:Double, bestLearningRate:Double, epochs:Int): Unit = {

    val bestModel = new SumOfWordVectorsModel(bestWordDim, bestVectorRegularizationStrength)
    StochasticGradientDescentLearner(bestModel, trainSetName, epochs, bestLearningRate, epochHook)

    println("Create word vector files.")
    val wordWriter = new PrintWriter(new File("./data/assignment3/word.txt" ))
    val paramWriter = new PrintWriter(new File("./data/assignment3/param.txt" ))
    var count = 0
    for ((paramName, paramBlock) <- bestModel.vectorParams if count < 2000) {
      println(s"$paramName:\n${paramBlock.param}\n")
  //    wordWriter.write(paramName + "\n")
      val predict = bestModel.predict(Seq(paramName))
      wordWriter.write(predict.compare(false) + "\n")

      val wordParam = paramBlock.param
      wordParam.foreach(param => {
        paramWriter.write(param+" ")
      })
      paramWriter.write("\n")
      count += 1
    }
    wordWriter.close()
    paramWriter.close()
    println("We have written "+bestModel.vectorParams.size+" words")
  }

def runGridSearch(wordDimSet:Range, vectorRegStrengthSet:NumericRange[Double], learningRateSet:NumericRange[Double], epochs:Int): Unit = {

  val gridSearchParams = mutable.MutableList[(Int, Double, Double, Double, Double)]()

  var isExploding = false
  val loop = new Breaks

  for (wordDim <- wordDimSet) {
    isExploding = false
    LookupTable.trainableWordVectors.clear()
      for (vectorRegStrength <- vectorRegStrengthSet) {
        if (isExploding) {
          println("Regularization Loop exploded gradients")
        }
        var isFirstRun = true
        loop.breakable {
          for (learningRate <- learningRateSet) {
            runSGDwithParam(wordDim, vectorRegStrength, learningRate, isFirstRun, epochs)
            isFirstRun = false
          }
        }
      }
  }

  if (gridSearchParams.isEmpty) {
    println("No best grid search parameters at that range!")
  } else {
    val bestDevAcc = gridSearchParams.maxBy(param => param._5)
    println("bestDevAcc=" + bestDevAcc)
  }

  def runSGDwithParam(wordDim:Int, vectorRegStrength:Double, learningRate:Double, isFirstRun:Boolean, epochs:Int):Unit = {
//    val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegStrength)
    val gridSearchModel = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegStrength, vectorRegStrength)
    StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, epochHook)
    println("wordDim %d\tvectorRegStrength %4.2f\tlearningRate %4.2f\t".format(wordDim, vectorRegStrength, learningRate))
    if (hasExplodingGradient(gridSearchModel)) {
      if (isFirstRun) isExploding = true
      loop.break()
      println("learningRateLoop exploded gradients")
    } else {
      val ratioOnTrainSet = 100 * Evaluator(gridSearchModel, trainSetName)
      val ratioOnValidSet = 100 * Evaluator(gridSearchModel, validationSetName)
      gridSearchParams.+=((wordDim, vectorRegStrength, learningRate, ratioOnTrainSet, ratioOnValidSet))
      println("ratioOnTrainSet %4.4f\tratioOnValidSet %4.4f\t".format(ratioOnTrainSet, ratioOnValidSet))
    }
    println()
    LookupTable.resetVectors()
  }
}


  /**
    * RNN Section
    */

//  val RNNmodel: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
//  StochasticGradientDescentLearner(RNNmodel, trainSetName, 100, learningRate, epochHook)

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