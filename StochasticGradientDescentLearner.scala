package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.FileWriter

import scala.util.control.Breaks

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {

  var previousDevLoss = Double.MaxValue

  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, isEarlyStop:Boolean, parentParams:String, logFileName:String): Unit = {
    println(parentParams)
    val epoch_loop = new Breaks
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)

    epoch_loop.breakable {
      for (i <- 0 until maxEpochs) {
        val startIterTime = System.nanoTime()
        var accLoss = 0.0
        for (j <- 0 until iterations) {
          val time = (System.nanoTime() - startIterTime) / 1e9
//          if (j % 10000 == 0) print(s"Iter %d Time: %2.2fs\r".format(j, time))
          val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
          val modelLoss = model.loss(sentence, target)
          val localLoss = modelLoss.forward()
          modelLoss.backward()
          modelLoss.update(learningRate) // updates parameters of the block
          accLoss += localLoss
          if (localLoss.isNaN) epoch_loop.break()
        }
        println(s"Epoch $i Train_Loss $accLoss")
        if (i % 1 == 0) {
          epochHook(i, model, epoch_loop, parentParams, accLoss, logFileName, isEarlyStop)
        }
      }
    }
    println()
  }

  def epochHook(iter: Int, model: Model, epoch_loop:Breaks, parentParams:String, trainLoss:Double, logFileName:String, isEarlyStop:Boolean): Unit = {
    val evaluatorOnValidSet = Evaluator(model, Main.validationSetName)
//    println("Epoch %4d\tDev_Loss %4.2f\tDev_Acc %4.2f".format(
//      iter, evaluatorOnValidSet._2, evaluatorOnValidSet._1))

    val devAcc = evaluatorOnValidSet._1
    val devLoss = evaluatorOnValidSet._2
    println(s"Epoch $iter $parentParams iter=$iter trainLoss=$trainLoss Dev_Loss=$devLoss Dev_Acc=$devAcc ")

    saveToFile(iter, trainLoss, evaluatorOnValidSet, parentParams, logFileName)

    // early stopping if loss on valid. set not going down
    if ((iter > 20) && evaluatorOnValidSet._2 > previousDevLoss && isEarlyStop) {
      epoch_loop.break()
      println("Break! " + evaluatorOnValidSet._2)
    }
    previousDevLoss = evaluatorOnValidSet._2
  }

  // write acc and loss to file in format: (iter train_acc train_loss valid_acc valid_loss)
  def saveToFile(iter: Int, trainLoss:Double, evaluatorOnValidSet: (Double, Double), parentParams:String, logFileName:String) = {
    val historyWriter = new FileWriter("./data/assignment3/" + logFileName, true)
    val devAcc = evaluatorOnValidSet._1
    val devLoss = evaluatorOnValidSet._2
    val outputToPrint = s"$parentParams iter=$iter trainLoss=$trainLoss Dev_Loss=$devLoss Dev_Acc=$devAcc "
    historyWriter.write(outputToPrint+"\n")
    historyWriter.close()
  }

}
