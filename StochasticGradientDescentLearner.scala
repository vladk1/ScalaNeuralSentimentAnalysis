package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.FileWriter

import uk.ac.ucl.cs.mr.statnlpbook.assignment3.SentimentAnalysisCorpus._

import scala.util.control.Breaks

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {

  var previousDevLoss = Double.MaxValue

  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, isEarlyStop:Boolean, parentParams:String, logFileName:String): Unit = {
    val epoch_loop = new Breaks
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    epoch_loop.breakable {
      for (i <- 0 until maxEpochs) {
        var accLoss = 0.0
        for (j <- 0 until iterations) {
//          if (j % 1000 == 0) print(s"Iter $j\r")
          val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
          val modelLoss = model.loss(sentence, target)
          val localLoss = modelLoss.forward()
          modelLoss.backward()
          modelLoss.update(learningRate) // updates parameters of the block
          accLoss += localLoss
          if (localLoss.isNaN) epoch_loop.break()
        }
//        if (i > 20) {
          epochHook(i, model, epoch_loop, parentParams, logFileName, isEarlyStop)
//        }
      }
    }
  }

  def epochHook(iter: Int, model: Model, epoch_loop:Breaks, parentParams:String, logFileName:String, isEarlyStop:Boolean): Unit = {
//    val evaluatorOnTrainSet = Evaluator(model, Main.trainSetName) // (accuracy percentage, loss)
    val evaluatorOnValidSet = Evaluator(model, Main.validationSetName)
    println("Epoch %4d\tDev_Loss %4.2f\tDev_Acc %4.2f".format(
      iter, evaluatorOnValidSet._2, evaluatorOnValidSet._1))

//    saveBestSetToFile(iter, evaluatorOnTrainSet, evaluatorOnValidSet, parentParams, logFileName)

    // early stopping if loss on valid. set not going down
    if ((iter > 20) && evaluatorOnValidSet._2 > previousDevLoss && isEarlyStop) {
      epoch_loop.break()
      println("Break! " + evaluatorOnValidSet._2)
    }
    previousDevLoss = evaluatorOnValidSet._2
  }

  // write acc and loss to file in format: (iter train_acc train_loss valid_acc valid_loss)
  def saveBestSetToFile(iter: Int, evaluatorOnTrainSet: (Double, Double), evaluatorOnValidSet: (Double, Double), parentParams:String, logFileName:String) = {
    val historyWriter = new FileWriter(getClass.getResourceAsStream("/bestres.txt") + logFileName, true)
    var outputToPrint = parentParams+" "+iter + " "
    for (evalTrain <- evaluatorOnTrainSet.productIterator.toList) {
      outputToPrint += (evalTrain.toString + " ")
    }
    for (evalValid <- evaluatorOnValidSet.productIterator.toList) {
      outputToPrint += (evalValid.toString + " ")
    }
    println(parentParams+"\n")
    historyWriter.write(outputToPrint+"\n")
    historyWriter.close()
  }

}
