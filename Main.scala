package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.collection.mutable
import scala.util.control.Breaks

/**
 * @author rockt
 */
object Main extends App {
  /**
   * Example training of a model
   *
   * Problems 2/3/4: perform a grid search over the parameters below
   */
//  val learningRate = 0.01
//  val vectorRegularizationStrength = 0.01
//  val matrixRegularizationStrength = 0.0
//  val wordDim = 10
//  val hiddenDim = 10

  val trainSetName = "train"
  val validationSetName = "dev"
  
//  val model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  //val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)

  def epochHook(iter: Int, accLoss: Double, model: Model): Unit = {
//    println("Epoch %4d\tLoss %8.4f\tTrain_Acc %4.2f\tDev_Acc %4.2f".format(
//      iter, accLoss, 100 * Evaluator(model, trainSetName), 100 * Evaluator(model, validationSetName)))
  }

  def hasExplodingGradient(model: Model): Boolean = {
    val (sentence, target) = SentimentAnalysisCorpus.getExample(trainSetName)
    val modelLoss = model.loss(sentence, target)
    modelLoss.forward().isNaN
  }

  def runSGDwithParam(wordDim:Int, vectorRegStrength:Double, learningRate:Double, isFirstRun:Boolean):Unit = {
    val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegStrength)
    StochasticGradientDescentLearner(gridSearchModel, trainSetName, 1, learningRate, epochHook)
    println("wordDim %d\tvectorRegStrength %4.2f\tlearningRate %4.2f\t".format(wordDim, vectorRegStrength, learningRate))
    if (hasExplodingGradient(gridSearchModel)) {
      if (isFirstRun) isExploding = true
      println("learningRateLoop exploded gradients")
      loop.break()
    } else {
      val ratioOnTrainSet = 100 * Evaluator(gridSearchModel, trainSetName)
      val ratioOnValidSet = 100 * Evaluator(gridSearchModel, validationSetName)
      gridSearchParams.+=((wordDim, vectorRegStrength, learningRate, ratioOnTrainSet, ratioOnValidSet))
      println("ratioOnTrainSet %4.2f\tratioOnValidSet %4.2f\t".format(ratioOnTrainSet, ratioOnValidSet))
    }
    println()
    LookupTable.resetVectors()
  }

//  StochasticGradientDescentLearner(model, trainSetName, 100, learningRate, epochHook)

  val wordDimSet = 4 to 10 by 1 // 1) 6
  val vectorRegStrengthSet =  0.4 to 0.8 by 0.05 // 1) 0.499
  val learningRateSet = 0.01 to 0.1 by 0.01 // 1) 0.05

//  1) not sure
//  2) 75.89


  val paramList = List(wordDimSet, vectorRegStrengthSet, learningRateSet)

  val gridSearchParams = mutable.MutableList[(Int, Double, Double, Double, Double)]()

  var isExploding = false
  val loop = new Breaks
  val regularizationLoop = new Breaks

  for (wordDim <- wordDimSet) {
    isExploding = false
    LookupTable.trainableWordVectors.clear()
    regularizationLoop.breakable {
      for (vectorRegStrength <- vectorRegStrengthSet) {
        if (isExploding) {
          println("Regularization Loop exploded gradients")
//          regularizationLoop.break()
        }
        loop.breakable {
          var isFirstRun = true
          for (learningRate <- learningRateSet) {
            runSGDwithParam(wordDim, vectorRegStrength, learningRate, isFirstRun)
            isFirstRun = false
          }
        }
      }
    }
  }
  if (gridSearchParams.isEmpty) {
    println("No best grid search parameters at that range!")
  } else {
    val bestDevAcc = gridSearchParams.maxBy(param => param._5)
    println("bestDevAcc="+bestDevAcc)
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