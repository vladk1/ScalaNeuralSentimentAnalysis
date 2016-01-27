package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.FileWriter

import scala.util.control.Breaks

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {

  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, parentParams:String): Unit = {
    val epoch_loop = new Breaks
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    epoch_loop.breakable {
      for (i <- 0 until maxEpochs) {
        var accLoss = 0.0
        for (j <- 0 until iterations) {
          if (j % 1000 == 0) print(s"Iter $j\r")
          val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
          val modelLoss = model.loss(sentence, target)
          val localLoss = modelLoss.forward()
          modelLoss.backward()
          modelLoss.update(learningRate) // updates parameters of the block
          accLoss += localLoss
          if (localLoss.isNaN) epoch_loop.break()
        }
//        if (i > 20) {
          epochHook(i, accLoss, model, epoch_loop, parentParams)
//        }
      }
    }
  }

  var previousDevLoss = Double.MaxValue

  def epochHook(iter: Int, accLoss: Double, model: Model, epoch_loop:Breaks, parentParams:String): Unit = {
    val evaluatorOnTrainSet = Evaluator(model, Main.trainSetName) // accuracy percentage, loss
    val evaluatorOnValidSet = Evaluator(model, Main.validationSetName)
    println("Epoch %4d\tLoss %8.2f\tTrain_Acc %4.2f\tDev_Acc %4.2f\t%4.2f\n".format(
      iter, accLoss, evaluatorOnTrainSet._1, evaluatorOnValidSet._1, evaluatorOnValidSet._2))

    saveBestSetToFile(iter, evaluatorOnTrainSet, evaluatorOnValidSet, parentParams)

    // early stopping if loss on valid. set not going down
    if ((iter > 20) && evaluatorOnValidSet._2 > previousDevLoss) {
      epoch_loop.break()
      println("Break! " + evaluatorOnValidSet._2)
    }

    previousDevLoss = evaluatorOnValidSet._2
  }

  // write acc and loss to file (iter train_acc train_loss valid_acc valid_loss)
  def saveBestSetToFile(iter: Int, evaluatorOnTrainSet: (Double, Double), evaluatorOnValidSet: (Double, Double), parentParams: String) = {
    val historyWriter = new FileWriter("./data/assignment3/param_history_rnn2.txt",true)
    historyWriter.write(parentParams+" "+iter + " ")
    // (accTrain accValid lossTrain lossValid)
    for (evalTrain <- evaluatorOnTrainSet.productIterator.toList) {
      historyWriter.write(evalTrain.toString + " ")
    }
    for (evalValid <- evaluatorOnValidSet.productIterator.toList) {
      historyWriter.write(evalValid.toString + " ")
    }
    historyWriter.write("\n")
    historyWriter.close()
  }

}
