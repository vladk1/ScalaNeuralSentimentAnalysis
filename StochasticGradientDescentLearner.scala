package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.FileWriter

import scala.util.control.Breaks

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {

  var previousDevLoss = Double.MaxValue

  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, isEarlyStop:Boolean, parentParams:String, logFileName:String): Unit = {
    val epoch_loop = new Breaks
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    val startIterTime = System.nanoTime()
    epoch_loop.breakable {
      for (i <- 0 until maxEpochs) {
        var accLoss = 0.0
        println(parentParams)

        for (j <- 0 until iterations) {
          val time = (System.nanoTime() - startIterTime) / 1e6
          if (j % 10000 == 0) print(s"Iter $j Time:$time\r")
          val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
          val modelLoss = model.loss(sentence, target)
          val localLoss = modelLoss.forward()
          modelLoss.backward()
          modelLoss.update(learningRate) // updates parameters of the block
          accLoss += localLoss
          if (localLoss.isNaN) epoch_loop.break()
        }

        println("epoch=" + i + " train_acc_loss=" + accLoss)
//        if (i > 10) {
//          val start_time = System.nanoTime()
          epochHook(i, model, epoch_loop, parentParams, logFileName, isEarlyStop)
//          val diff = (System.nanoTime() - start_time) / 1e6
//          println("timePerHook="+diff+"\n")
//        }
      }
    }
    val time = (System.nanoTime() - startIterTime) / 1e6
    println("grid search step time = "+time)
    println()
  }

  def epochHook(iter: Int, model: Model, epoch_loop:Breaks, parentParams:String, logFileName:String, isEarlyStop:Boolean): Unit = {
    val evaluatorOnValidSet = Evaluator(model, Main.validationSetName)
    println("Epoch %4d\tDev_Loss %4.2f\tDev_Acc %4.2f".format(
      iter, evaluatorOnValidSet._2, evaluatorOnValidSet._1))

//    saveBestSetToFile(iter, evaluatorOnTrainSet, evaluatorOnValidSet, parentParams, logFileName)

    // early stopping if loss on valid. set not going down
    if ((iter > 50) && evaluatorOnValidSet._2 > previousDevLoss && isEarlyStop) {
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
