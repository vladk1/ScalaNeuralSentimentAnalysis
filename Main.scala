package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{File, PrintWriter}

import scala.collection.mutable
import scala.util.Random
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

//    Using for RNN (single run)
  val learningRate = 0.01 // tried 0.01, 0.1, 1.0, 10.0, 1.0,
  val vectorRegularizationStrength = 0.01 // tried 0.01, 0.1, 0.01, _, _, 5.0
  val matrixRegularizationStrength = 0.0 // tried 0.01, 0.1, 0.01, _, _, 5.0
  val wordDim = 10 // tried 10, 5
  val hiddenDim = 5 // tried 10, 5

  val trainSetName = "train"
  val validationSetName = "dev"

  val wordDimSet = 5 to 13 by 1
  val vectorRegStrengthSet = (-15 to 1 by 1).map(a => Math.pow(10,a)) // in case Nan - higher regularizer
  // in case Nan - lower learning rate (hence we can stop iterating after reaching Nan in this loop)
  val learningRateSet = (-6 to 1 by 1).map(a => Math.pow(10,a))

  //  1) 75.43  (6, 0.45, 0.06) with 1 iteration:
  //  2) 77.16  (7, 0.1, 0.05) -//-
  //  3) 76.298 (5, 0.05, 0.045) -//-
  //  4) 75.433 (6, 0.01, 0.015) with better tokenizer (gives us 225 998 words vs old 275 826)
  //  5) 75.099 (7, 0.01, 0.01) 10 iter
  //  6) 75.433 (6, 0.001, 0.001) 10 iter
  //  7) 77.230 (10,0.001, 0.031) 1 iter
  // iterating through log
  // 8) 73.702  (10,1.0E-6,0.01) 1 iter
  // 9) 74.7 (7, 1.0E-8, 0.1) 1 iter
  //10) 75.89 (7,1.0E-10,0.001) 10 iter
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
//  val SGDLmodel = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  StochasticGradientDescentLearner(SGDLmodel, trainSetName, 100, learningRate, epochHook)

//    runGridSearch(wordDimSet, vectorRegStrengthSet, learningRateSet, 1)
//  ToDo we can visualize parameter space i.e. 3d graph of params and validation set

//  prints vectorparams to file
//  writeVectorsFromBestModelToFile(7, 0.1, 0.05, 10)

  def epochHook(iter: Int, accLoss: Double, model: Model): Unit = {
    println("Epoch %4d\tLoss %8.2f\tTrain_Acc %4.2f\tDev_Acc %4.2f".format(
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

    val params: Array[(String,VectorParam)] = bestModel.vectorParams.toArray
    val rnd = new Random()

    while (count < 2000) {
      val example = params(rnd.nextInt(params.length))
      val paramName = example._1
      val paramBlock = example._2
      println(s"$paramName:\n${paramBlock.param}\n")
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


def runGridSearch(wordDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {

  val gridSearchParams = mutable.MutableList[(Int, Double, Double, Double, Double)]()

  val loop = new Breaks

  for (wordDim <- wordDimSet) {
    for (vectorRegStrength <- vectorRegStrengthSet) {
      loop.breakable {
        for (learningRate <- learningRateSet) {
          runSGDwithParam(wordDim, vectorRegStrength, learningRate, epochs)
        }
      }
    }
  }

  if (gridSearchParams.isEmpty) {
    println("No best grid search parameters at that range!")
  } else {
    val bestDevAcc = gridSearchParams.maxBy(param => param._5)
    println("bestDevAcc=" + bestDevAcc)
    val historyWriter = new PrintWriter(new File("./data/assignment3/param_history.txt" ))
    for (param <- gridSearchParams) {
      val data = param._1 + " " + param._2 + " " + param._3 + " " + param._5
      historyWriter.write(data+"\n")
    }
    historyWriter.close()
  }


  def runSGDwithParam(wordDim:Int, vectorRegStrength:Double, learningRate:Double, epochs:Int):Unit = {
    val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegStrength)
    //    val gridSearchModel = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegStrength, vectorRegStrength)
    StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate, epochHook)
    println("wordDim %d\tvectorRegStrength %4.10f\tlearningRate %4.10f\t".format(wordDim, vectorRegStrength, learningRate))
//    if (hasExplodingGradient(gridSearchModel)) {
//      loop.break() // don't increase learning rate, as we already have exploding gradients
//      println("learningRateLoop exploded gradients")
//    } else {
      val ratioOnTrainSet = 100 * Evaluator(gridSearchModel, trainSetName)
      val ratioOnValidSet = 100 * Evaluator(gridSearchModel, validationSetName)
      gridSearchParams.+=((wordDim, vectorRegStrength, learningRate, ratioOnTrainSet, ratioOnValidSet))
      println("ratioOnTrainSet %4.2f\tratioOnValidSet %4.2f\t".format(ratioOnTrainSet, ratioOnValidSet))
//    }
    println()
    LookupTable.resetVectors()
    LookupTable.trainableWordVectors.clear()
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